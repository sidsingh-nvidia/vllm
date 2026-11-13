# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-ins for the Megatron-LM helpers the vendored MInf kernels import."""


def null_decorator(*args, **kwargs):
    """No-op decorator used when triton is unavailable (mirrors megatron.core.utils)."""
    if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
        return args[0]

    def inner(func):
        return func

    return inner
