"""
LangChain Tool Bridge - Wraps MCP tools as LangChain BaseTool instances.

Converts ToolExecutor tools (MCPTool) into LangChain-compatible BaseTool
subclasses so they can be used in a LangGraph agent. This is a thin wrapper
that delegates execution to the ToolExecutor; concerns like namespace
injection and Korrel8r normalization belong in the agent layer, not here.

Usage:
    from chatbots.tools import create_langchain_tools
    from chatbots.tool_executor import ToolExecutor

    tool_executor: ToolExecutor = ...
    lc_tools = create_langchain_tools(tool_executor)
    # lc_tools is a List[BaseTool] ready for a LangGraph agent
"""

from typing import Any, Dict, List, Optional, Type

from enum import Enum
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import BaseTool

from chatbots.tool_executor import ToolExecutor, MCPTool
from common.pylogger import get_python_logger

logger = get_python_logger()

# Mapping from JSON Schema type strings to Python types used when
# dynamically building Pydantic models from MCP tool input schemas.
_JSON_SCHEMA_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_pydantic_model(
    name: str, schema: Dict[str, Any]
) -> Type[BaseModel]:
    """Build a Pydantic model from a JSON Schema ``properties`` dict.

    Only top-level properties are converted; nested object schemas are
    represented as plain ``dict`` fields.  This keeps the bridge simple
    while still giving LangChain the typed args_schema it needs.

    Args:
        name: Name used for the generated Pydantic model class.
        schema: The ``input_schema`` dict from an MCPTool (JSON Schema).

    Returns:
        A dynamically created Pydantic model class.
    """
    properties: Dict[str, Any] = schema.get("properties", {})
    required_fields: set = set(schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        field_type: Any = _JSON_SCHEMA_TYPE_MAP.get(json_type, Any)

        # For array types, use List[<element_type>] so the generated
        # Pydantic schema includes the ``items`` field.  Google Gemini
        # rejects tool schemas that declare ``type: "array"`` without it.
        if json_type == "array":
            items_schema = field_schema.get("items", {})
            item_type = _JSON_SCHEMA_TYPE_MAP.get(
                items_schema.get("type", "string"), Any
            )
            field_type = List[item_type]

        # Preserve enum constraints so providers (especially Gemini)
        # know the valid values for string parameters.
        enum_values = field_schema.get("enum")
        if enum_values and field_type is str:
            enum_cls = Enum(  # type: ignore[misc]
                f"{name}_{field_name}_enum",
                {v: v for v in enum_values},
            )
            field_type = enum_cls

        description = field_schema.get("description", "")
        default_value = field_schema.get("default")
        is_required = field_name in required_fields

        if is_required:
            field_definitions[field_name] = (
                field_type,
                Field(description=description),
            )
        else:
            # Preserve the schema's default value instead of always using None.
            # This gives models (especially Gemini) a hint about expected format.
            field_definitions[field_name] = (
                Optional[field_type],
                Field(default=default_value, description=description),
            )

    return create_model(name, **field_definitions)


class MCPToolWrapper(BaseTool):
    """LangChain BaseTool wrapper around a single MCP tool.

    Routes execution through the injected ``ToolExecutor`` so that the
    underlying MCP infrastructure handles the actual work.
    """

    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    # Private attribute — not part of the Pydantic schema exposed to LangChain
    _tool_executor: ToolExecutor

    def __init__(
        self,
        mcp_tool: MCPTool,
        tool_executor: ToolExecutor,
        **kwargs: Any,
    ) -> None:
        # Build the dynamic args_schema before calling super().__init__
        # so LangChain sees a proper Pydantic model for argument validation.
        schema_model = _json_schema_to_pydantic_model(
            f"{mcp_tool.name}_args", mcp_tool.input_schema or {}
        )

        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=schema_model,
            **kwargs,
        )
        # Store the executor after super().__init__ to avoid Pydantic
        # attempting to validate it as a model field.
        object.__setattr__(self, "_tool_executor", tool_executor)

    def _run(self, **kwargs: Any) -> str:
        """Execute the MCP tool synchronously via the ToolExecutor.

        Args:
            **kwargs: Tool arguments matching the ``args_schema``.

        Returns:
            Tool result as a string.
        """
        # Strip None values so the underlying tool only receives
        # arguments that were explicitly provided.
        arguments = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(
            "Executing MCP tool '%s' via LangChain bridge with args: %s",
            self.name,
            arguments,
        )

        try:
            result = self._tool_executor.call_tool(self.name, arguments)
            logger.info(
                "MCP tool '%s' returned result (length: %d)",
                self.name,
                len(result) if result else 0,
            )
            return result
        except Exception as e:
            logger.error("Error executing MCP tool '%s': %s", self.name, e)
            return f"Error executing {self.name}: {str(e)}"


def create_langchain_tools(
    tool_executor: ToolExecutor,
    allowlist: Optional[set] = None,
) -> List[BaseTool]:
    """Create LangChain BaseTool instances from all available MCP tools.

    Args:
        tool_executor: The ToolExecutor that provides tool metadata and
            handles execution.
        allowlist: Optional set of tool names to include. When provided,
            only tools whose names appear in this set are returned.
            Pass ``None`` (the default) to include all tools.

    Returns:
        A list of LangChain BaseTool instances ready for use in a
        LangGraph agent.
    """
    mcp_tools = tool_executor.list_tools()
    lc_tools: List[BaseTool] = []

    for mcp_tool in mcp_tools:
        if allowlist is not None and mcp_tool.name not in allowlist:
            continue

        try:
            wrapper = MCPToolWrapper(
                mcp_tool=mcp_tool,
                tool_executor=tool_executor,
            )
            lc_tools.append(wrapper)
        except Exception as e:
            logger.error(
                "Failed to create LangChain wrapper for tool '%s': %s",
                mcp_tool.name,
                e,
            )

    tool_names = [t.name for t in lc_tools]
    logger.info(
        "Created %d LangChain tool wrappers: %s",
        len(lc_tools),
        ", ".join(tool_names) if tool_names else "(none)",
    )

    return lc_tools
