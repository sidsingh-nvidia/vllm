# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Megatron-Core inference's NVLS MoE dispatcher as a vLLM all2all backend.

See manager.py. The Triton kernels in this package are vendored from
NVIDIA Megatron-LM (megatron/core/inference); see each file's header.
"""

from .manager import NvlinkAgRsAll2AllManager

__all__ = ["NvlinkAgRsAll2AllManager"]
