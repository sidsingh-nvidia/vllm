# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU correctness test of the nvlink_allgather_reducescatter dispatch / combine (no vLLM engine).

    torchrun --nproc-per-node 4 tools/nvlink_agrs_plugin/test_nvlink_agrs_dispatch.py

Loads the nvlink_allgather_reducescatter package through the plugin loader (so it also works against
an installed vLLM that lacks it) and drives NvlinkAgRsAll2AllManager with a stub
communication group. For random, unequal per-rank token counts (zero included)
it checks:

  * dispatch: every gathered tensor (bf16 hidden, fp32 topk weights, int32 topk
    ids, plus fp32 scales and a 1-D int32 extra whose rows need padding) equals
    a torch.distributed all_gather reference, bit for bit;
  * combine: with each rank computing partials only for its local experts
    (expert e scales the row by (e + 1)), the reduce-scatter-v result equals the
    locally computed sum_k w_k * (id_k + 1) * h, in bf16 and fp32 combine;
  * the once-per-forward metadata path (two "layers" under one forward context);
  * CUDA graph capture and replay of dispatch + expert stub + combine.

Prints one PASS/FAIL line per case on rank 0 and exits non-zero on failure.
"""

import os
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault(
    "VLLM_NVLINK_AGRS_SRC",
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "vllm", "distributed",
                 "device_communicators"),
)
from vllm_nvlink_agrs_plugin import _load_nvlink_allgather_reducescatter  # noqa: E402

minf = _load_nvlink_allgather_reducescatter()
from vllm.forward_context import override_forward_context  # noqa: E402

PER_RANK_MAX = 512
NUM_EXPERTS = 64


class StubManager(minf.NvlinkAgRsAll2AllManager):
    """NvlinkAgRsAll2AllManager without vLLM's parallel state / config."""

    def __init__(self, group, combine_dtype="auto"):
        # Skip All2AllManagerBase.__init__ (needs vLLM's parallel groups).
        self.per_rank_max = PER_RANK_MAX
        self.combine_dtype_mode = combine_dtype
        self.metadata_every_layer = False
        self._max_blocks_env = os.environ.get("VLLM_NVLINK_AGRS_MAX_BLOCKS")
        self._states = {}
        self.dp_world_size = dist.get_world_size(group)
        self.comm = SimpleNamespace(
            unique_name="test", device_group=group,
            world_size=dist.get_world_size(group), rank_in_group=dist.get_rank(group),
        )
        self.sizes = None

    def _get_comm_group(self, is_sequence_parallel):
        return self.comm

    def _get_sizes(self, num_local_tokens, comm_group):
        return self.sizes


def ref_all_gather(t, sizes):
    """Variable-size all_gather via padding to max."""
    ws = len(sizes)
    m = max(sizes)
    pad = torch.zeros((m,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    pad[: t.shape[0]] = t
    outs = [torch.empty_like(pad) for _ in range(ws)]
    dist.all_gather(outs, pad)
    return torch.cat([o[:s] for o, s in zip(outs, sizes)], dim=0)


def make_inputs(local, hidden, topk, gen):
    dev = torch.cuda.current_device()
    h = torch.randn(local, hidden, generator=gen, device="cpu").to(dev, torch.bfloat16)
    w = torch.rand(local, topk, generator=gen, device="cpu").to(dev)
    ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=gen)[:topk] for _ in range(local)]
    ).to(dev, torch.int32) if local else torch.empty(0, topk, dtype=torch.int32, device=dev)
    scales = torch.rand(local, (hidden + 127) // 128, generator=gen).to(dev)
    lora = torch.randint(0, 7, (local,), generator=gen, dtype=torch.int32).to(dev)
    return h, w, ids, scales, lora


def expert_stub(gh, gw, gids, rank, ws, out):
    """Partial MoE output from this rank's experts: sum_k [e local] w_k (e+1) h."""
    per = NUM_EXPERTS // ws
    local_mask = (gids >= rank * per) & (gids < (rank + 1) * per)
    coef = (gw * (gids.float() + 1.0) * local_mask).sum(dim=1, keepdim=True)
    out.copy_((gh.float() * coef).to(out.dtype))
    return out


def expected_combine(h, w, ids, ws, partial_dtype):
    """Reference: per-rank partials rounded to partial_dtype, then summed in fp32."""
    per = NUM_EXPERTS // ws
    acc = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
    for r in range(ws):
        m = (ids >= r * per) & (ids < (r + 1) * per)
        coef = (w * (ids.float() + 1.0) * m).sum(dim=1, keepdim=True)
        acc += (h.float() * coef).to(partial_dtype).float()
    return acc


def run_case(mgr, hidden, topk, combine, gen, rank, ws, layers=1, use_ctx=False):
    mgr.combine_dtype_mode = combine
    # Same random counts on every rank (shared seed); rank 1 always gets 0 in
    # one case below via the generator draw.
    counts = torch.randint(0, PER_RANK_MAX + 1, (ws,), generator=gen).tolist()
    if gen.initial_seed() % 3 == 0:
        counts[1 % ws] = 0
    local = counts[rank]
    mgr.sizes = counts
    torch.manual_seed(1000 + rank)
    lgen = torch.Generator().manual_seed(1234 + 17 * rank + hidden + topk)
    errors = []
    ctx = override_forward_context(SimpleNamespace()) if use_ctx else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for layer in range(layers):
            h, w, ids, scales, lora = make_inputs(local, hidden, topk, lgen)
            gh, gw, gids, extras = mgr.dispatch(h, w, ids, extra_tensors=[scales, lora])
            gscales, glora = extras
            total = sum(counts)
            for name, got, mine in (("hidden", gh, h), ("weights", gw, w), ("ids", gids, ids),
                                    ("scales", gscales, scales), ("lora", glora, lora)):
                ref = ref_all_gather(mine, counts)
                if got.shape != ref.shape or not torch.equal(got, ref):
                    errors.append(f"L{layer} dispatch {name} mismatch "
                                  f"{tuple(got.shape)} vs {tuple(ref.shape)}")
            buf = mgr.allocate_combine_input((total, hidden), torch.bfloat16, h.device)
            if combine in ("auto", "bf16"):
                if buf is None:
                    errors.append("allocate_combine_input returned None for bf16")
            partial = buf if buf is not None else torch.empty(
                total, hidden, dtype=torch.bfloat16, device=h.device)
            expert_stub(gh, gw, gids, rank, ws, partial)
            out = torch.empty(local, hidden, dtype=torch.bfloat16, device=h.device)
            mgr.combine_into_output(partial, out)
            exp = expected_combine(h, w, ids, ws, torch.bfloat16)
            if local:
                err = (out.float() - exp).abs().max().item()
                tol = 2e-2 * max(1.0, exp.abs().max().item())
                if err > tol:
                    errors.append(f"L{layer} combine err {err:.4g} > tol {tol:.4g}")
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return counts, errors


def run_graph_case(mgr, hidden, topk, rank, ws, gen):
    """Capture dispatch + stub experts + combine in a CUDA graph, replay with new data."""
    mgr.combine_dtype_mode = "auto"
    counts = torch.randint(1, PER_RANK_MAX + 1, (ws,), generator=gen).tolist()
    mgr.sizes = counts
    local, total = counts[rank], sum(counts)
    lgen = torch.Generator().manual_seed(99 + rank)
    h, w, ids, _, _ = make_inputs(local, hidden, topk, lgen)
    out = torch.empty(local, hidden, dtype=torch.bfloat16, device=h.device)

    def step():
        gh, gw, gids = mgr.dispatch(h, w, ids)
        part = mgr.allocate_combine_input((total, hidden), torch.bfloat16, h.device)
        expert_stub(gh, gw, gids, rank, ws, part)
        mgr.combine_into_output(part, out)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()  # warmup: allocate + compile outside capture
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    errors = []
    for it in range(3):
        nh, nw, nids, _, _ = make_inputs(local, hidden, topk, lgen)
        h.copy_(nh)
        w.copy_(nw)
        ids.copy_(nids)
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        exp = expected_combine(h, w, ids, ws, torch.bfloat16)
        err = (out.float() - exp).abs().max().item()
        tol = 2e-2 * max(1.0, exp.abs().max().item())
        if err > tol:
            errors.append(f"replay {it}: combine err {err:.4g} > tol {tol:.4g}")
    dist.barrier()
    return counts, errors


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    group = dist.new_group(list(range(ws)))
    mgr = StubManager(group)
    if rank == 0:
        print(f"world_size={ws} device={torch.cuda.get_device_name()} "
              f"torch={torch.__version__}", flush=True)

    cases = []
    seed = 0
    for hidden in (2688, 4096, 1024):
        for topk in (1, 6, 22):
            for combine in ("auto", "fp32"):
                cases.append(dict(hidden=hidden, topk=topk, combine=combine))
    # Repeat a few with once-per-forward metadata over two "layers".
    ctx_cases = [dict(hidden=2688, topk=6, combine="auto"),
                 dict(hidden=4096, topk=22, combine="fp32")]

    n_pass = n_fail = 0

    def report(label, counts, errors):
        nonlocal n_pass, n_fail
        flags = torch.tensor([len(errors)], device="cuda")
        dist.all_reduce(flags)
        ok = flags.item() == 0
        n_pass += ok
        n_fail += not ok
        if errors:
            print(f"[rank {rank}] {label}: " + "; ".join(errors[:4]), flush=True)
        dist.barrier()
        if rank == 0:
            print(f"{'PASS' if ok else 'FAIL'} {label} counts={counts}", flush=True)

    for c in cases:
        seed += 1
        gen = torch.Generator().manual_seed(seed)
        label = f"hidden={c['hidden']} topk={c['topk']} combine={c['combine']}"
        try:
            counts, errors = run_case(mgr, c["hidden"], c["topk"], c["combine"], gen, rank, ws)
        except Exception:
            counts, errors = None, [traceback.format_exc()]
        report(label, counts, errors)

    for c in ctx_cases:
        seed += 1
        gen = torch.Generator().manual_seed(seed)
        label = (f"fwd-ctx 2 layers hidden={c['hidden']} topk={c['topk']} "
                 f"combine={c['combine']}")
        try:
            counts, errors = run_case(mgr, c["hidden"], c["topk"], c["combine"], gen, rank, ws,
                                      layers=2, use_ctx=True)
        except Exception:
            counts, errors = None, [traceback.format_exc()]
        report(label, counts, errors)

    for hidden, topk in ((2688, 6), (4096, 22)):
        seed += 1
        gen = torch.Generator().manual_seed(seed)
        try:
            counts, errors = run_graph_case(mgr, hidden, topk, rank, ws, gen)
        except Exception:
            counts, errors = None, [traceback.format_exc()]
        report(f"cudagraph hidden={hidden} topk={topk}", counts, errors)

    if rank == 0:
        print(f"SUMMARY: {n_pass} passed, {n_fail} failed", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
