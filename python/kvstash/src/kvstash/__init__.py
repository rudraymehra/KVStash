"""KVStash brand alias: re-exports the kvblockd client.

`pip install KVStash` pulls the real client package (kvblockd); the
daemon is a single Go binary — install it per the repo README.
"""

from kvblockd.client import Client

__all__ = ["Client"]
