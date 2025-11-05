"""Tool Executor Interface for Chatbots

This interface allows chatbots to execute MCP tools without depending
on specific implementations (UI client vs direct MCP server).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ToolExecutor(ABC):
    """Interface for executing MCP tools.

    Implementations can use:
    - MCPToolExecutor (for direct server-side access)
    - UIToolExecutor (for UI/remote access via HTTP/SSE) - if needed in future
    - Mock executor (for testing)
    """

    @abstractmethod
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools with their schemas.

        Returns:
            List of tool definitions in format:
            [
                {
                    "name": "tool_name",
                    "description": "What the tool does",
                    "input_schema": {...}  # JSON Schema
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dict

        Returns:
            Tool execution result as string

        Raises:
            Exception: If tool execution fails
        """
        pass
