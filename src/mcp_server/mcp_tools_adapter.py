"""
MCP Server Adapter for Chatbots

This module provides an adapter that implements MCPToolsInterface for use in the MCP server process.
It wraps the ObservabilityMCPServer instance to provide tool calling functionality to chatbots.
"""

import asyncio
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbots.mcp_tools_interface import MCPToolsInterface, MCPTool
from common.pylogger import get_python_logger

logger = get_python_logger()


class MCPServerAdapter(MCPToolsInterface):
    """Adapter for calling MCP tools directly in the MCP server process.

    This adapter wraps an ObservabilityMCPServer instance and provides
    the MCPToolsInterface for chatbots to call tools.
    """

    def __init__(self, mcp_server):
        """Initialize with ObservabilityMCPServer instance.

        Args:
            mcp_server: ObservabilityMCPServer instance that provides the MCP tools
        """
        self.mcp_server = mcp_server
        logger.info("🔌 MCPServerAdapter initialized for direct tool access")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool directly via the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            Tool result as a string

        Raises:
            Exception: If tool execution fails
        """
        try:
            logger.info(f"🔧 MCPServerAdapter calling tool: {tool_name}")

            # Get the tool (needs to be awaited)
            tool = asyncio.run(self.mcp_server.mcp.get_tool(tool_name))

            # Execute the tool (needs to be awaited)
            result = asyncio.run(tool.run(arguments))

            # Extract text from result
            if hasattr(result, 'content'):
                # Handle FastMCP ToolResult object
                if isinstance(result.content, list):
                    # Content is a list of content blocks
                    result_text = ""
                    for block in result.content:
                        if hasattr(block, 'text'):
                            result_text += block.text
                        elif isinstance(block, dict) and 'text' in block:
                            result_text += block['text']
                        else:
                            result_text += str(block)
                    return result_text
                else:
                    return str(result.content)
            elif isinstance(result, str):
                return result
            else:
                return str(result)

        except Exception as e:
            logger.error(f"❌ Error calling tool {tool_name}: {e}")
            raise

    def list_tools(self) -> List[MCPTool]:
        """List all available MCP tools from the server.

        Returns:
            List of MCPTool objects with metadata
        """
        try:
            logger.info("📋 MCPServerAdapter listing available tools")

            # Access the tool manager from FastMCP
            tool_manager = self.mcp_server.mcp._tool_manager

            mcp_tools = []
            for tool_name, tool_info in tool_manager._tools.items():
                # Extract schema from the tool function
                schema = tool_info.get('schema', {})

                # Create MCPTool object
                mcp_tool = MCPTool(
                    name=tool_name,
                    description=schema.get('description', ''),
                    input_schema=schema.get('inputSchema', {})
                )
                mcp_tools.append(mcp_tool)

            logger.info(f"✅ MCPServerAdapter found {len(mcp_tools)} tools")
            return mcp_tools

        except Exception as e:
            logger.error(f"❌ Error listing tools: {e}")
            return []
