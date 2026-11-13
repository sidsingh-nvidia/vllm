# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE EP dispatch / combine with Megatron-Core inference's NVLS collectives.

``all2all_backend="nvlink_allgather_reducescatter"`` is the allgather_reducescatter backend with its
NCCL all-gather-v / reduce-scatter-v swapped for the multimem (NVLink SHARP)
kernels of Megatron-Core inference's ``NVLSAllGatherVDispatcher``:

* dispatch: every rank multicast-stores its local rows (hidden states, topk
  weights, topk ids and any extra tensors such as quantization scales) into a
  fixed-size symmetric buffer of ``max_tokens_per_rank * world_size`` rows, at
  the row offset given by the exclusive prefix sum of the per-rank token counts;
  the experts then run on the first ``sum(counts)`` rows.
* combine: each rank's expert partials ``[sum(counts), hidden]`` sit in a
  symmetric buffer and every rank reads its own rows with ``multimem.ld_reduce``,
  which sums the partials of all ranks in the NVSwitch.

The per-rank counts are exchanged on the GPU once per forward pass by a single
CTA metadata kernel (``fused_metadata_update``) that writes
``[total, rank_offset, max_count]`` into a fixed device tensor. The collective
kernels read the offset and the max from that tensor, launch a grid that depends
only on rank-invariant values, and take the local count from the input shape,
so dispatch and combine have no host sync, no allocation and fixed addresses:
they can be captured into (piecewise or full) CUDA graphs.

Everything else (prepare/finalize, sequence-parallel MoE, which experts may be
used) is inherited from allgather_reducescatter: the host-side token counts that
vLLM already keeps in the forward context's DP metadata size the returned views.

Requirements: Hopper or newer GPUs, every rank of the EP group in one NVLink
domain, torch symmetric memory with multicast support, triton.

Environment knobs (all optional):

* ``VLLM_NVLINK_AGRS_MAX_TOKENS_PER_RANK``: rows each rank may contribute per
  dispatch; sizes the buffers. Default: ``max_num_batched_tokens``.
* ``VLLM_NVLINK_AGRS_COMBINE_DTYPE``: ``bf16`` | ``fp32`` | ``auto`` (default).
  ``auto`` reduces in the experts' output dtype and lets them write straight
  into the symmetric buffer; ``fp32`` copies the partials to an fp32 buffer and
  reduces in fp32 (Megatron's ``inference_moe_combine_precision='fp32'``).
* ``VLLM_NVLINK_AGRS_MAX_BLOCKS``: grid cap of the collectives. Default: as
  Megatron (``ep_collectives_max_num_blocks``: 4 blocks/SM up to EP 8, else 1).
* ``VLLM_NVLINK_AGRS_METADATA_EVERY_LAYER=1``: refresh the token-count metadata
  in every MoE layer instead of once per forward pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from ..all2all import AgRsAll2AllManager
from .kernel_config import SYMM_MEM_KERNEL_CONFIG, ep_collectives_max_num_blocks
from .metadata import fused_metadata_update
from .variable_collectives import (
    multimem_all_gather_v,
    multimem_all_gatherv_2tensor,
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)

logger = init_logger(__name__)

_ALIGN = 16


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _ensure_signal_pad_size(world_size: int) -> None:
    """Grow torch's symmetric-memory signal pad to fit the collectives' barriers.

    Same rule as Megatron's SymmetricMemoryBuffer: one int32 slot per (block,
    peer); torch reads the size at allocation time, so call it before each
    allocation. Older torch has a fixed pad and no setter.
    """
    import torch.distributed._symmetric_memory as symm_mem

    if not hasattr(symm_mem, "set_signal_pad_size"):
        return
    needed = SYMM_MEM_KERNEL_CONFIG.signal_pad_bytes(world_size)
    if symm_mem.get_signal_pad_size() < needed:
        symm_mem.set_signal_pad_size(needed)


def _symm_alloc(nbytes: int, group: dist.ProcessGroup) -> tuple[torch.Tensor, Any]:
    """Collectively allocate a uint8 symmetric buffer and rendezvous it."""
    import torch.distributed._symmetric_memory as symm_mem

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "nvlink_allgather_reducescatter: a symmetric buffer would be allocated during CUDA graph "
            "capture. Buffers are created on first use, which must happen in an "
            "eager forward (e.g. the profile run); this dispatch has a tensor "
            "shape the eager runs never produced."
        )
    if hasattr(symm_mem, "enable_symm_mem_for_group"):
        try:
            symm_mem.enable_symm_mem_for_group(group.group_name)
        except Exception:  # deprecated / no-op on newer torch
            pass
    _ensure_signal_pad_size(group.size())
    buf = symm_mem.empty(_round_up(max(nbytes, _ALIGN), _ALIGN), dtype=torch.uint8,
                         device="cuda")
    hdl = symm_mem.rendezvous(buf, group)
    if hdl is None or not getattr(hdl, "multicast_ptr", 0):
        raise RuntimeError(
            "nvlink_allgather_reducescatter: symmetric memory has no multicast support on this group; "
            "the NVLS dispatcher needs Hopper+ GPUs in one NVLink domain."
        )
    return buf, hdl


@dataclass
class _GroupState:
    """Buffers and per-step metadata of one communication group."""

    device_group: dist.ProcessGroup
    rank: int
    world_size: int
    per_rank_max: int
    # [3] int32: [total tokens, this rank's row offset, max tokens over ranks].
    step_metadata: torch.Tensor
    meta_buf: torch.Tensor
    meta_hdl: Any
    # (role, row_bytes) -> (uint8 buffer, handle)
    buffers: dict = field(default_factory=dict)
    last_ctx: Any = None
    last_local_tokens: int = -1

    @property
    def global_max(self) -> int:
        return self.per_rank_max * self.world_size

    def rank_token_offset(self) -> torch.Tensor:
        return self.step_metadata[1:2]

    def ep_max_tokens(self) -> torch.Tensor:
        return self.step_metadata[2:3]

    def buffer(self, role: str, row_bytes: int) -> tuple[torch.Tensor, Any]:
        key = (role, row_bytes)
        if key not in self.buffers:
            self.buffers[key] = _symm_alloc(row_bytes * self.global_max, self.device_group)
            logger.info(
                "nvlink_allgather_reducescatter: allocated symmetric buffer %s: %d rows x %d B = %.1f MiB",
                key, self.global_max, row_bytes,
                row_bytes * self.global_max / 2**20,
            )
        return self.buffers[key]


class NvlinkAgRsAll2AllManager(AgRsAll2AllManager):
    """allgather_reducescatter with Megatron-Core inference's NVLS AG-v / RS-v."""

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        vllm_config = get_current_vllm_config()
        env_max = os.environ.get("VLLM_NVLINK_AGRS_MAX_TOKENS_PER_RANK")
        if env_max:
            self.per_rank_max = int(env_max)
        else:
            max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
            parallel_config = vllm_config.parallel_config
            if parallel_config.use_sequence_parallel_moe:
                # Each TP rank dispatches its ceil(n / tp) slice of the DP rank's tokens.
                tp = parallel_config.tensor_parallel_size
                max_tokens = (max_tokens + tp - 1) // tp
            self.per_rank_max = max_tokens
        self.combine_dtype_mode = os.environ.get(
            "VLLM_NVLINK_AGRS_COMBINE_DTYPE", "auto").lower()
        assert self.combine_dtype_mode in ("auto", "bf16", "fp32"), (
            f"VLLM_NVLINK_AGRS_COMBINE_DTYPE={self.combine_dtype_mode!r}; "
            "expected auto, bf16 or fp32"
        )
        self.metadata_every_layer = (
            os.environ.get("VLLM_NVLINK_AGRS_METADATA_EVERY_LAYER", "0") == "1")
        self._max_blocks_env = os.environ.get("VLLM_NVLINK_AGRS_MAX_BLOCKS")
        self._states: dict[str, _GroupState] = {}
        logger.info(
            "Using nvlink_allgather_reducescatter all2all manager (Megatron-Core inference NVLS "
            "AllGather-V / ReduceScatter-V), max %d tokens per rank, combine "
            "dtype %s.", self.per_rank_max, self.combine_dtype_mode,
        )

    # ── State ─────────────────────────────────────────────────────────────────

    def _state(self, comm_group: Any) -> _GroupState:
        key = comm_group.unique_name
        st = self._states.get(key)
        if st is not None:
            return st
        device_group = comm_group.device_group
        world_size = comm_group.world_size
        if not self._states:
            # Set once, before the first symmetric allocation: signal pads are
            # sized from it (see kernel_config.py) and it must match on all ranks.
            if self._max_blocks_env:
                max_blocks = int(self._max_blocks_env)
            else:
                num_sms = torch.cuda.get_device_properties(
                    torch.cuda.current_device()).multi_processor_count
                max_blocks = ep_collectives_max_num_blocks(num_sms, world_size)
            SYMM_MEM_KERNEL_CONFIG.max_num_blocks = max_blocks
        meta_buf, meta_hdl = _symm_alloc(world_size * 4, device_group)
        st = _GroupState(
            device_group=device_group,
            rank=comm_group.rank_in_group,
            world_size=world_size,
            per_rank_max=self.per_rank_max,
            step_metadata=torch.zeros(3, dtype=torch.int32, device="cuda"),
            meta_buf=meta_buf[: world_size * 4].view(torch.int32),
            meta_hdl=meta_hdl,
        )
        assert meta_hdl.rank == st.rank and meta_hdl.world_size == world_size
        self._states[key] = st
        return st

    def _update_metadata(self, st: _GroupState, local_tokens: int) -> None:
        """Exchange token counts once per forward pass (every call with no context)."""
        if local_tokens > st.per_rank_max:
            raise RuntimeError(
                f"nvlink_allgather_reducescatter: {local_tokens} local tokens exceed the buffer capacity "
                f"of {st.per_rank_max} per rank; raise "
                "VLLM_NVLINK_AGRS_MAX_TOKENS_PER_RANK."
            )
        ctx = get_forward_context() if is_forward_context_available() else None
        if (ctx is not None and ctx is st.last_ctx and not self.metadata_every_layer):
            # All MoE layers of one forward see the same local count. A mismatch
            # cannot be fixed locally (the update is collective), so fail loudly.
            if local_tokens != st.last_local_tokens:
                raise RuntimeError(
                    f"nvlink_allgather_reducescatter: token count changed within a forward pass "
                    f"({st.last_local_tokens} -> {local_tokens}); set "
                    "VLLM_NVLINK_AGRS_METADATA_EVERY_LAYER=1."
                )
            return
        fused_metadata_update(
            local_tokens=local_tokens,
            local_buf=st.meta_buf,
            symm_mem_hdl=st.meta_hdl,
            step_metadata=st.step_metadata,
        )
        # Strong reference: an id() could be reused by the next context.
        st.last_ctx = ctx
        st.last_local_tokens = local_tokens

    # ── Gather ────────────────────────────────────────────────────────────────

    def _gather(
        self, st: _GroupState, tensors: list[torch.Tensor], total: int, role: str
    ) -> list[torch.Tensor]:
        """All-gather-v each [local, ...] tensor into [total, ...] symmetric views."""
        local = tensors[0].shape[0]
        prepared = []  # (input2d, out2d_full, orig_shape_tail, orig_cols)
        for i, t in enumerate(tensors):
            assert t.shape[0] == local, "all dispatched tensors need the same rows"
            tail = tuple(t.shape[1:])
            cols = 1
            for d in tail:
                cols *= d
            t2 = t.reshape(local, cols)
            esize = t2.element_size()
            row_bytes = cols * esize
            if row_bytes % 8 != 0:
                # The NVLS kernels move 64/128-bit packs; pad narrow rows.
                padded_cols = _round_up(row_bytes, _ALIGN) // esize
                t2 = torch.nn.functional.pad(t2, (0, padded_cols - cols))
                row_bytes = padded_cols * esize
            if not t2.is_contiguous() or t2.data_ptr() % _ALIGN != 0:
                t2 = t2.contiguous() if not t2.is_contiguous() else t2.clone()
            buf, hdl = st.buffer(f"{role}{i}", row_bytes)
            out = buf[: row_bytes * st.global_max].view(t2.dtype).view(
                st.global_max, t2.shape[1])
            prepared.append((t2, out, hdl, tail, cols))

        common = dict(
            rank_token_offset=st.rank_token_offset(),
            ep_max_tokens=st.ep_max_tokens(),
            per_rank_max_tokens=st.per_rank_max,
        )
        i = 0
        while i < len(prepared):
            n = min(3, len(prepared) - i)
            grp = prepared[i:i + n]
            if n == 3:
                multimem_all_gatherv_3tensor(
                    grp[0][1], grp[1][1], grp[2][1],
                    grp[0][0], grp[1][0], grp[2][0],
                    grp[0][2], grp[1][2], grp[2][2], **common)
            elif n == 2:
                multimem_all_gatherv_2tensor(
                    grp[0][1], grp[1][1], grp[0][0], grp[1][0],
                    grp[0][2], grp[1][2], **common)
            else:
                multimem_all_gather_v(grp[0][1], grp[0][0], grp[0][2], **common)
            i += n

        results = []
        for t2, out, _, tail, cols in prepared:
            res = out[:total]
            if res.shape[1] != cols:
                res = res[:, :cols].contiguous()
            results.append(res.view(total, *tail))
        return results

    def _sizes_and_state(self, local: int, is_sequence_parallel: bool):
        comm_group = self._get_comm_group(is_sequence_parallel)
        if comm_group.world_size == 1:
            return comm_group, None, local
        sizes = self._get_sizes(local, comm_group)
        assert sizes[comm_group.rank_in_group] == local, (sizes, local)
        st = self._state(comm_group)
        self._update_metadata(st, local)
        return comm_group, st, int(sum(sizes))

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        _, st, total = self._sizes_and_state(hidden_states.shape[0], is_sequence_parallel)
        if st is None:
            return super().dispatch_router_logits(
                hidden_states, router_logits, is_sequence_parallel, extra_tensors)
        tensors = [hidden_states, router_logits] + list(extra_tensors or [])
        out = self._gather(st, tensors, total, role="rl")
        if extra_tensors is not None:
            return out[0], out[1], out[2:]
        return out[0], out[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        _, st, total = self._sizes_and_state(hidden_states.shape[0], is_sequence_parallel)
        if st is None:
            return super().dispatch(hidden_states, topk_weights, topk_ids,
                                    is_sequence_parallel, extra_tensors)
        tensors = [hidden_states, topk_weights, topk_ids] + list(extra_tensors or [])
        out = self._gather(st, tensors, total, role="d")
        if extra_tensors is None:
            return out[0], out[1], out[2]
        return out[0], out[1], out[2], out[3:]

    # ── Combine ───────────────────────────────────────────────────────────────

    def _combine_dtype(self, dtype: torch.dtype) -> torch.dtype:
        if self.combine_dtype_mode == "fp32":
            return torch.float32
        if self.combine_dtype_mode == "bf16":
            return torch.bfloat16
        return dtype

    def _rsv(self, st: _GroupState, hidden: int, dtype: torch.dtype):
        esize = torch.empty((), dtype=dtype).element_size()
        row_bytes = hidden * esize
        assert row_bytes % 16 == 0, (
            f"nvlink_allgather_reducescatter combine needs 16-byte rows; hidden={hidden} {dtype}")
        buf, hdl = st.buffer("rsv", row_bytes)
        return buf[: row_bytes * st.global_max].view(dtype).view(st.global_max, hidden), hdl

    def allocate_combine_input(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor | None:
        comm_group = self._get_comm_group(is_sequence_parallel)
        if comm_group.world_size == 1 or len(shape) != 2:
            return None
        if dtype not in (torch.bfloat16, torch.float32):
            return None
        if self._combine_dtype(dtype) != dtype:
            return None
        st = self._state(comm_group)
        if shape[0] > st.global_max or (shape[1] * dtype.itemsize) % 16 != 0:
            return None
        rsv, _ = self._rsv(st, shape[1], dtype)
        return rsv[: shape[0]]

    def combine_into_output(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        comm_group = self._get_comm_group(is_sequence_parallel)
        if comm_group.world_size == 1:
            return super().combine_into_output(hidden_states, output, is_sequence_parallel)
        st = self._state(comm_group)
        total, hidden = hidden_states.shape
        local = output.shape[0]
        assert local == st.last_local_tokens or not self._dispatched(st), (
            f"nvlink_allgather_reducescatter combine: output has {local} rows, dispatch had "
            f"{st.last_local_tokens}")
        rdtype = self._combine_dtype(hidden_states.dtype)
        rsv, hdl = self._rsv(st, hidden, rdtype)
        if not (hidden_states.dtype == rdtype
                and hidden_states.data_ptr() == rsv.data_ptr()
                and hidden_states.is_contiguous()):
            rsv[:total].copy_(hidden_states)
        direct = (output.dtype == rdtype and output.is_contiguous()
                  and output.data_ptr() % _ALIGN == 0)
        out = output if direct else torch.empty(
            (local, hidden), dtype=rdtype, device=output.device)
        multimem_reduce_scatter_v(
            out.view(local, hidden),
            rsv,
            hdl,
            rank_token_offset=st.rank_token_offset(),
            ep_max_tokens=st.ep_max_tokens(),
            per_rank_max_tokens=st.per_rank_max,
        )
        if not direct:
            output.copy_(out.view(output.shape))
        return output

    @staticmethod
    def _dispatched(st: _GroupState) -> bool:
        return st.last_local_tokens >= 0

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        comm_group = self._get_comm_group(is_sequence_parallel)
        if comm_group.world_size == 1:
            return super().combine(hidden_states, is_sequence_parallel)
        st = self._state(comm_group)
        assert self._dispatched(st), "nvlink_allgather_reducescatter combine before any dispatch"
        output = torch.empty(
            (st.last_local_tokens, hidden_states.shape[1]),
            dtype=hidden_states.dtype, device=hidden_states.device)
        return self.combine_into_output(hidden_states, output, is_sequence_parallel)

    def destroy(self):
        self._states.clear()
