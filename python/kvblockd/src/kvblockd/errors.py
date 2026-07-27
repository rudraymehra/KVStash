"""Typed error hierarchy — no bare socket exceptions leak past the client.

ONE root on purpose: `except KvblockdError` must catch every client-raised
error, including the two most common operational ones (StatusError,
FrameError) — they lived in protocol.py on plain Exception and a caller
following this module's own advice silently missed them. protocol.py
re-exports both for compatibility; the pool imports from HERE only (the
error taxonomy, not the wire codec, is its dependency).
"""

from __future__ import annotations


class KvblockdError(Exception):
    """Base for every client-raised error."""


class ConnectionLost(KvblockdError):
    """The socket closed or errored mid-exchange; the connection is dead and
    must be evicted from the pool (never re-pooled)."""


class FatalProtocol(KvblockdError):
    """The server sent an F_FATAL frame (magic/version/CRC violation, §9).
    The connection is closed by the server after this."""


class FrameError(KvblockdError):
    """Malformed frame on the wire (bad magic/version/CRC or truncation, or a
    body too short for its declared layout). The stream state is unknown —
    the connection must be evicted."""


class StatusError(KvblockdError):
    """A non-OK verb status. The connection stays in sync (the caller may
    re-pool it) — distinct from FrameError, which desyncs the stream."""

    def __init__(self, op: int, status: int):
        self.op = op
        self.status = status
        super().__init__(f"op {int(op):#x}: status {self._status_name(status)}")

    @staticmethod
    def _status_name(status: int) -> str:
        # Deferred import: protocol imports THIS module at load time, so the
        # name lookup must not create an import-time cycle. By the time a
        # StatusError is raised, protocol is always fully loaded.
        try:
            from kvblockd.protocol import Status

            return Status(status).name if status in Status._value2member_map_ else hex(status)
        except Exception:  # noqa: BLE001 — a pretty name must never mask the real error
            return hex(status)
