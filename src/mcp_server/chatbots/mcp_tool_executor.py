"""MCP Tool Executor - Direct Server-Side Implementation

This executor calls tools directly via ObservabilityMCPServer.
Use this when chatbots run server-side (e.g., in /v1/chat endpoint).
"""

import asyncio
import nest_asyncio
from typing import Dict, Any, List

from .tool_executor import ToolExecutor
from common.pylogger import get_python_logger

logger = get_python_logger()

# Allow nested event loops (needed for FastAPI + asyncio.run())
nest_asyncio.apply()


class MCPToolExecutor(ToolExecutor):
    """Execute tools directly via ObservabilityMCPServer.

    This implementation is for server-side usage where we have
    direct access to the MCP server instance.
    """

    def __init__(self, server):
        """Initialize with MCP server instance.

        Args:
            server: ObservabilityMCPServer instance
        """
        self.server = server

    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get tools from MCP server dynamically.

        Fetches tool definitions directly from FastMCP server,
        eliminating the need for hard-coded tool lists.

        Returns:
            List of tool definitions with name, description, and schema
        """
        try:
            # Get tools dict from FastMCP server
            tools_dict = await self.server.mcp.get_tools()

            tool_list = []
            for name, tool_obj in tools_dict.items():
                # Convert FastMCP tool to MCP protocol format
                mcp_tool = tool_obj.to_mcp_tool()

                tool_list.append({
                    "name": mcp_tool.name,
                    "description": mcp_tool.description or "",
                    "input_schema": mcp_tool.inputSchema or {}
                })

            logger.info(f"Fetched {len(tool_list)} tools from MCP server")
            return tool_list
        except Exception as e:
            logger.error(f"Error fetching tools from MCP server: {e}")
            return []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool via MCP server (direct in-process call).

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result as string

        Raises:
            Exception: If tool execution fails
        """
        logger.info(f"🔨 MCPToolExecutor executing tool: {tool_name}")
        logger.info(f"📥 Tool arguments: {arguments}")

        try:
            # Define async function to get and run tool
            async def run_tool():
                # Get the tool function from FastMCP (this is async)
                logger.info(f"⚙️ Getting tool '{tool_name}' from MCP server...")
                tool = await self.server.mcp.get_tool(tool_name)
                logger.info(f"▶️ Running tool '{tool_name}'...")
                # Run the tool
                return await tool.run(arguments)

            # Run the tool (nest_asyncio allows this even in async context)
            result = asyncio.run(run_tool())
            logger.info(f"📤 Raw tool result type: {type(result)}, value: {str(result)[:500]}")

            # Convert result to string
            # FastMCP tools now return ToolResult objects
            if hasattr(result, 'content'):
                # ToolResult object with content attribute
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from first content item
                    if isinstance(content[0], dict) and 'text' in content[0]:
                        final_result = content[0]['text']
                        logger.info(f"✅ Extracted text from ToolResult content dict, length: {len(final_result)}")
                        return final_result
                    elif hasattr(content[0], 'text'):
                        final_result = content[0].text
                        logger.info(f"✅ Extracted text from ToolResult content object, length: {len(final_result)}")
                        return final_result
                    final_result = str(content[0])
                    logger.info(f"✅ Converted ToolResult content[0] to string, length: {len(final_result)}")
                    return final_result
                final_result = str(content)
                logger.info(f"✅ Converted ToolResult content to string, length: {len(final_result)}")
                return final_result
            elif isinstance(result, list) and len(result) > 0:
                # Legacy: list of content items
                if isinstance(result[0], dict) and 'text' in result[0]:
                    final_result = result[0]['text']
                    logger.info(f"✅ Extracted text from dict, length: {len(final_result)}")
                    return final_result
                final_result = str(result[0])
                logger.info(f"✅ Converted result[0] to string, length: {len(final_result)}")
                return final_result

            final_result = str(result)
            logger.info(f"✅ Converted full result to string, length: {len(final_result)}")
            return final_result

        except Exception as e:
            logger.error(f"❌ Error executing tool {tool_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error executing {tool_name}: {str(e)}"
