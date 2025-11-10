"""
MCP Client Adapter

This module provides an adapter that wraps MCPClientHelper to implement
the ToolClientInterface. This allows chatbots to use MCP tools without
having a direct dependency on mcp_client_helper.

This is the Adapter pattern: it adapts MCPClientHelper's interface
to match the ToolClientInterface that chatbots expect.
"""

import logging
import os
import sys
from typing import List, Dict, Any

# Add parent path to import chatbots module
_PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_PATH not in sys.path:
    sys.path.insert(0, _PARENT_PATH)

from chatbots.tool_client_interface import ToolClientInterface
from mcp_client_helper import MCPClientHelper

logger = logging.getLogger(__name__)


class MCPClientAdapter(ToolClientInterface):
    """Adapter that wraps MCPClientHelper to implement ToolClientInterface.

    This adapter allows MCPClientHelper to be used with chatbots without
    chatbots having a direct dependency on mcp_client_helper.

    The adapter implements the ToolClientInterface by delegating calls
    to the underlying MCPClientHelper instance.
    """

    def __init__(self, mcp_client: MCPClientHelper = None):
        """Initialize adapter with MCPClientHelper instance.

        Args:
            mcp_client: MCPClientHelper instance to wrap.
                       If None, creates a new instance.
        """
        self.mcp_client = mcp_client or MCPClientHelper()
        logger.info("MCPClientAdapter initialized")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server.

        Delegates to MCPClientHelper.get_available_tools().

        Returns:
            List of tool definitions with name, description, and input_schema
        """
        try:
            tools = self.mcp_client.get_available_tools()
            logger.debug(f"MCPClientAdapter: Retrieved {len(tools)} tools")
            return tools
        except Exception as e:
            logger.error(f"MCPClientAdapter: Error getting tools: {e}")
            raise

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool via MCP client.

        Delegates to MCPClientHelper.call_tool_sync().

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dictionary

        Returns:
            Tool execution result from MCP server
        """
        try:
            logger.debug(f"MCPClientAdapter: Calling tool '{tool_name}' with args: {arguments}")
            result = self.mcp_client.call_tool_sync(tool_name, arguments)
            logger.debug(f"MCPClientAdapter: Tool '{tool_name}' completed")
            return result
        except Exception as e:
            logger.error(f"MCPClientAdapter: Error calling tool '{tool_name}': {e}")
            raise
