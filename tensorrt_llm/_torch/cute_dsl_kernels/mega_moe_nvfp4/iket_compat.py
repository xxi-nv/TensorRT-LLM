# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for optional ``cutlass.cute.iket`` support."""

# IKET (In-Kernel Event Tracing) markers are only available in cutlass-dsl
# wheels that ship the ``iket`` dialect. Functional tests do not need the
# dialect, so fall back to no-op markers when the import is unavailable.
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket

    class _IketShim:
        """No-op IKET shim used when the dialect is not available."""

        @staticmethod
        def range_push(_name, *_args, **_kwargs):
            return None

        @staticmethod
        def range_pop(*_args, **_kwargs):
            return None

        @staticmethod
        def range_start(_name, *_args, **_kwargs):
            return None

        @staticmethod
        def range_end(_token=None, *_args, **_kwargs):
            return None

        @staticmethod
        def mark(_name, *_args, **_kwargs):
            return None

    iket = _IketShim()  # type: ignore
