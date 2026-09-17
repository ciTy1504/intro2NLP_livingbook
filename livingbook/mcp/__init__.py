"""Model Context Protocol integration.

Two independent directions:

  server.py  exposes the Living Book to MCP clients (Claude Desktop, Claude Code)
             over stdio, so the system can be driven conversationally.
  client.py  mounts external MCP servers as Living Book tools, so the tool layer is
             extensible without code changes.

Both use stdio. Nothing here needs to be hosted: the state this system operates on —
SQLite, the manuscript, the key pool, the git repo — is local, so a remote server
could not reach it anyway.
"""

from .client import MCPToolBridge, get_bridge, mount_external_servers, probe_servers

__all__ = ["MCPToolBridge", "mount_external_servers", "get_bridge", "probe_servers"]
