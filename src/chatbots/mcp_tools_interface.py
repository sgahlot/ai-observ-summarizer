"""
MCP Tools Interface - Abstract interface for chatbots to call tools.

This interface allows chatbots to call MCP tools without depending on
whether they're running in the MCP server process or in a client process.

Implementations:
- MCPServerAdapter (mcp_server/mcp_tools_adapter.py) - Direct tool calls in server
- MCPClientAdapter (ui/mcp_client_adapter.py) - HTTP calls to MCP server from UI
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class MCPTool:
    """Represents an MCP tool with metadata."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MCPToolsInterface(ABC):
    """Abstract interface for calling MCP tools.

    This allows chatbots to call tools without knowing whether they're
    running in the MCP server process or as a client.
    """

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result as a string.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool

        Returns:
            Tool result as string

        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        pass

    @abstractmethod
    def list_tools(self) -> List[MCPTool]:
        """List all available MCP tools.

        Returns:
            List of MCPTool objects
        """
        pass

    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get metadata for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            MCPTool object if found, None otherwise
        """
        pass
