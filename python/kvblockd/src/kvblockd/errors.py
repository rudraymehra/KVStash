"""Typed error hierarchy — no bare socket exceptions leak past the client."""

from __future__ import annotations


class KvblockdError(Exception):
    """Base for every client-raised error."""


class ConnectionLost(KvblockdError):
    """The socket closed or errored mid-exchange; the connection is dead and
    must be evicted from the pool (never re-pooled)."""


class FatalProtocol(KvblockdError):
    """The server sent an F_FATAL frame (magic/version/CRC violation, §9).
    The connection is closed by the server after this."""
