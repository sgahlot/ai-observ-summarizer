"""
MCP Client Adapter

This module provides an adapter that wraps MCPClientHelper to implement
the ToolExecutor interface. This allows chatbots to use MCP tools without
having a direct dependency on mcp_client_helper.

This is the Adapter pattern: it adapts MCPClientHelper's interface
to match the ToolExecutor interface that chatbots expect.
"""

import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Add parent path to import chatbots module
_PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_PATH not in sys.path:
    sys.path.insert(0, _PARENT_PATH)

from chatbots.tool_executor import ToolExecutor, MCPTool
from mcp_client_helper import MCPClientHelper

logger = logging.getLogger(__name__)


class MCPClientAdapter(ToolExecutor):
    """Adapter that wraps MCPClientHelper to implement ToolExecutor.

    This adapter allows MCPClientHelper to be used with chatbots without
    chatbots having a direct dependency on mcp_client_helper.

    The adapter implements the ToolExecutor interface by delegating calls
    to the underlying MCPClientHelper instance.
    """

    def __init__(self, mcp_client: MCPClientHelper = None):
        """Initialize adapter with MCPClientHelper instance.

        Args:
            mcp_client: MCPClientHelper instance to wrap.
                       If None, creates a new instance.
        """
        self.mcp_client = mcp_client or MCPClientHelper()
        self._tools_cache: Optional[List[MCPTool]] = None
        logger.info("MCPClientAdapter initialized")

    def list_tools(self) -> List[MCPTool]:
        """List all available MCP tools from the server via client.

        Returns:
            List of MCPTool objects with metadata
        """
        try:
            # Use cached tools if available
            if self._tools_cache is not None:
                return self._tools_cache

            logger.debug("MCPClientAdapter: Listing available MCP tools")

            # Get tools from MCP client helper
            tools_data = self.mcp_client.get_available_tools()

            mcp_tools = []
            for tool_info in tools_data:
                mcp_tool = MCPTool(
                    name=tool_info.get('name', ''),
                    description=tool_info.get('description', ''),
                    input_schema=tool_info.get('input_schema', {})
                )
                mcp_tools.append(mcp_tool)

            # Cache the tools
            self._tools_cache = mcp_tools

            logger.debug(f"MCPClientAdapter: Retrieved {len(mcp_tools)} tools")
            return mcp_tools

        except Exception as e:
            logger.error(f"MCPClientAdapter: Error getting tools: {e}")
            raise

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get metadata for a specific MCP tool.

        Args:
            tool_name: Name of the tool

        Returns:
            MCPTool object if found, None otherwise
        """
        try:
            # Get all tools (uses cache if available)
            mcp_tools = self.list_tools()

            # Find the specific tool
            for mcp_tool in mcp_tools:
                if mcp_tool.name == tool_name:
                    logger.debug(f"Found tool: {tool_name}")
                    return mcp_tool

            logger.warning(f"Tool not found: {tool_name}")
            return None

        except Exception as e:
            logger.error(f"MCPClientAdapter: Error getting tool {tool_name}: {e}")
            return None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool via MCP client and return result as string.

        Delegates to MCPClientHelper.call_tool_sync().

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dictionary

        Returns:
            Tool execution result as string
        """
        try:
            logger.debug(f"MCPClientAdapter: Calling tool '{tool_name}' with args: {arguments}")
            result = self.mcp_client.call_tool_sync(tool_name, arguments)
            logger.debug(f"MCPClientAdapter: Tool '{tool_name}' completed")

            # Extract text from result
            if result and len(result) > 0:
                result_text = result[0].get('text', '')
                return result_text
            else:
                return f"No results returned from {tool_name}"

        except Exception as e:
            logger.error(f"MCPClientAdapter: Error calling tool '{tool_name}': {e}")
            raise
