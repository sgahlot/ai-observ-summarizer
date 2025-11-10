"""
Tool Client Interface

This module defines the abstract interface for tool discovery and execution.
Any client (UI, CLI, API, etc.) can implement this interface to provide
tool capabilities to chatbots.

This enables dependency inversion: chatbots depend on this interface,
not on concrete implementations like mcp_client_helper.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ToolClientInterface(ABC):
    """Abstract interface for tool discovery and execution.

    Any client (UI, CLI, API, etc.) must implement this interface
    to provide tool capabilities to chatbots.

    This allows chatbots to be client-agnostic and work with any
    tool provider that implements this interface.
    """

    @abstractmethod
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from the tool provider.

        Returns:
            List of tool definitions, where each tool is a dict with:
            - name (str): Tool name
            - description (str): Tool description
            - input_schema (dict): JSON schema for tool parameters

        Example:
            [
                {
                    'name': 'execute_promql',
                    'description': 'Execute a PromQL query against Prometheus',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'start_time': {'type': 'string'},
                            'end_time': {'type': 'string'}
                        },
                        'required': ['query']
                    }
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dictionary

        Returns:
            Tool execution result. Format depends on implementation,
            but typically a list of dicts with 'text' key:
            [{'type': 'text', 'text': 'result content'}]

        Raises:
            Exception: If tool execution fails

        Example:
            result = client.call_tool(
                'execute_promql',
                {'query': 'up', 'start_time': '1h'}
            )
            # Returns: [{'type': 'text', 'text': 'query results...'}]
        """
        pass
