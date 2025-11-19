"""
MCP Server Adapter for Chatbots

This module provides an adapter that implements ToolExecutor for use in the MCP server process.
It wraps the ObservabilityMCPServer instance to provide tool execution functionality to chatbots.
"""

import asyncio
from typing import Dict, Any, List, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbots.tool_executor import ToolExecutor, MCPTool
from common.pylogger import get_python_logger

logger = get_python_logger()


class MCPServerAdapter(ToolExecutor):
    """Adapter for executing MCP tools directly in the MCP server process.

    This adapter wraps an ObservabilityMCPServer instance and provides
    the ToolExecutor interface for chatbots to execute tools.
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

            # Get the current event loop if we're already in one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create a new one
                loop = None

            if loop is not None:
                # We're already in an async context, use run_coroutine_threadsafe
                # or directly call the tool function
                import concurrent.futures
                import threading

                # Run in a new thread with its own event loop
                result_future = concurrent.futures.Future()

                def run_in_thread():
                    try:
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            tool = new_loop.run_until_complete(self.mcp_server.mcp.get_tool(tool_name))
                            result = new_loop.run_until_complete(tool.run(arguments))
                            result_future.set_result(result)
                        finally:
                            new_loop.close()
                    except Exception as e:
                        result_future.set_exception(e)

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                result = result_future.result()
            else:
                # No running loop, use asyncio.run
                tool = asyncio.run(self.mcp_server.mcp.get_tool(tool_name))
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

            logger.info(f"✅ MCPServerAdapter found {len(mcp_tools)} MCP tools")
            return mcp_tools

        except Exception as e:
            logger.error(f"❌ Error listing MCP tools: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get metadata for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            MCPTool object if found, None otherwise
        """
        try:
            logger.info(f"🔍 MCPServerAdapter getting tool: {tool_name}")

            # Access the tool manager from FastMCP
            tool_manager = self.mcp_server.mcp._tool_manager

            # Get the tool info
            tool_info = tool_manager._tools.get(tool_name)
            if not tool_info:
                logger.warning(f"⚠️ Tool not found: {tool_name}")
                return None

            # Extract schema from the tool function
            schema = tool_info.get('schema', {})

            # Create MCPTool object
            mcp_tool = MCPTool(
                name=tool_name,
                description=schema.get('description', ''),
                input_schema=schema.get('inputSchema', {})
            )

            logger.info(f"✅ MCPServerAdapter found tool: {tool_name}")
            return mcp_tool

        except Exception as e:
            logger.error(f"❌ Error getting tool {tool_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
