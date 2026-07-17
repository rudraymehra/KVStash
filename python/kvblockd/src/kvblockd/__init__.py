"""kvblockd — Python client for the KVB1 KV-cache block store."""

from kvblockd.protocol import FrameError, Op, Status, StatusError

__all__ = ["Op", "Status", "StatusError", "FrameError"]
__version__ = "0.1.0"
