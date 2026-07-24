"""FastMCP wrapper for the Node REPL MCP server.
Requires: pip install fastmcp
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
import node_repl as nr

mcp = FastMCP("Node REPL")

# Proxy core tool functions to FastMCP tools
mcp.tool()(nr.js)
mcp.tool()(nr.js_add_node_module_dir)
mcp.tool()(nr.js_reset)

if __name__ == "__main__":
    mcp.run()

