# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM general plugin: the nvlink_allgather_reducescatter MoE all2all backend on an unmodified vLLM.

For a prebuilt vLLM (e.g. the v0.32.0 container) that lacks
``all2all_backend="nvlink_allgather_reducescatter"``. Put this directory on PYTHONPATH (its
``.dist-info`` registers the ``vllm.general_plugins`` entry point, no pip
install needed), then run with::

    VLLM_NVLINK_AGRS=1 vllm serve ... --enable-expert-parallel \
        --all2all-backend allgather_reducescatter

vLLM loads general plugins in every process (API server, engine core, each
worker) before the device communicators are built. With ``VLLM_NVLINK_AGRS=1``
this plugin loads the ``nvlink_allgather_reducescatter`` package from the source tree next to it
(or from ``VLLM_NVLINK_AGRS_SRC``, the directory holding ``nvlink_allgather_reducescatter/``) as
``vllm.distributed.device_communicators.nvlink_agrs`` and replaces
``all2all.AgRsAll2AllManager`` with ``NvlinkAgRsAll2AllManager``, a subclass, so
everything that vLLM wires for allgather_reducescatter (prepare/finalize,
sequence-parallel MoE, expert selection) stays as is and only the two
collectives change. Without ``VLLM_NVLINK_AGRS=1`` it does nothing.
"""

import importlib.util
import os
import sys

_PKG = "vllm.distributed.device_communicators.nvlink_agrs"


def _default_src() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # <repo>/tools/nvlink_agrs_plugin/vllm_nvlink_agrs_plugin -> <repo>
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(repo, "vllm", "distributed", "device_communicators")


def _load_nvlink_allgather_reducescatter():
    if _PKG in sys.modules:
        return sys.modules[_PKG]
    try:
        # A vLLM built from the branch ships the package itself.
        return importlib.import_module(_PKG)
    except ImportError:
        pass
    pkg_dir = os.path.join(os.environ.get("VLLM_NVLINK_AGRS_SRC") or _default_src(), "nvlink_agrs")
    init = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init):
        raise ImportError(f"nvlink_allgather_reducescatter plugin: {init} not found; set VLLM_NVLINK_AGRS_SRC")
    spec = importlib.util.spec_from_file_location(
        _PKG, init, submodule_search_locations=[pkg_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_PKG, None)
        raise
    return mod


def register() -> None:
    if os.environ.get("VLLM_NVLINK_AGRS", "0") != "1":
        return
    from vllm.distributed.device_communicators import all2all
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    if getattr(all2all, "_nvlink_allgather_reducescatter_patched", False):
        return
    # Import first: NvlinkAgRsAll2AllManager subclasses the original AgRsAll2AllManager.
    manager_cls = _load_nvlink_allgather_reducescatter().NvlinkAgRsAll2AllManager
    all2all.AgRsAll2AllManager = manager_cls
    all2all._nvlink_allgather_reducescatter_patched = True
    logger.info(
        "nvlink_allgather_reducescatter plugin: --all2all-backend allgather_reducescatter now runs the "
        "Megatron-Core inference NVLS AllGather-V / ReduceScatter-V (%s).",
        sys.modules[_PKG].__file__,
    )
