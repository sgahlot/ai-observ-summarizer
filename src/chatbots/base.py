"""
Base Chat Bot - Common Functionality

This module provides the base class for DeterministicChatBot (and any future
rule-based bots). The LangGraph agent path does NOT inherit from this class;
it uses the standalone utility functions in langchain_agent.py instead.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable

from chatbots.tool_executor import ToolExecutor
from common.pylogger import get_python_logger

logger = get_python_logger()

# PromQL functions, keywords, and common label names that should NOT get
# namespace injected — only actual metric names should.  Module-level constant
# to avoid rebuilding on every call.
_PROMQL_SKIP = frozenset({
    # Aggregation operators
    'sum', 'avg', 'min', 'max', 'count', 'stddev', 'stdvar',
    'topk', 'bottomk', 'quantile', 'count_values',
    # Rate/counter functions
    'rate', 'irate', 'increase', 'delta', 'idelta',
    # Histogram functions
    'histogram_quantile', 'histogram_count', 'histogram_sum',
    # Label functions
    'label_replace', 'label_join',
    # Math functions
    'abs', 'ceil', 'floor', 'round', 'clamp', 'clamp_min', 'clamp_max',
    'exp', 'ln', 'log2', 'log10', 'sqrt',
    # Sort/utility functions
    'sort', 'sort_desc', 'time', 'timestamp',
    'vector', 'scalar', 'sgn',
    # Range functions
    'changes', 'resets', 'deriv', 'predict_linear',
    'absent', 'absent_over_time', 'present_over_time',
    # Date functions
    'day_of_month', 'day_of_week', 'days_in_month',
    'hour', 'minute', 'month', 'year',
    # Keywords and operators
    'by', 'without', 'on', 'ignoring', 'group_left', 'group_right',
    'bool', 'offset', 'and', 'or', 'unless',
    # Over-time functions
    'avg_over_time', 'min_over_time', 'max_over_time',
    'sum_over_time', 'count_over_time', 'stddev_over_time',
    'last_over_time', 'quantile_over_time',
    # Common Kubernetes label names (appear in by/without/grouping clauses)
    'pod', 'namespace', 'container', 'node', 'instance', 'job',
    'service', 'deployment', 'daemonset', 'statefulset', 'replicaset',
    'phase', 'reason', 'condition', 'type', 'resource', 'unit',
    'device', 'interface', 'mode', 'cpu', 'endpoint', 'alertname',
    'alertstate', 'severity', 'le', 'model_name', 'model',
    # Numeric literals used in comparisons (won't match our regex, but safe)
    'inf', 'nan',
})


class BaseChatBot(ABC):
    """Base class for rule-based chat bot implementations (e.g., DeterministicChatBot)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        tool_executor: ToolExecutor = None
    ):
        if tool_executor is None:
            raise ValueError(
                "tool_executor is required. Pass a MCPServerAdapter instance "
                "from the MCP server context"
            )

        if not isinstance(tool_executor, ToolExecutor):
            raise TypeError(
                f"tool_executor must implement ToolExecutor, got {type(tool_executor)}"
            )

        self.model_name = model_name
        self.api_key = api_key if api_key is not None else self._get_api_key()
        self.tool_executor = tool_executor

        from core.config import NAMESPACE_AWARE_TOOLS
        self._namespace_aware_tools = NAMESPACE_AWARE_TOOLS

        logger.info(f"{self.__class__.__name__} initialized with model: {self.model_name}")

    @abstractmethod
    def _get_api_key(self) -> Optional[str]:
        """Get API key for this bot implementation."""

    def _get_max_tool_result_length(self) -> int:
        """Maximum length for tool results before truncation. Default: 5000 chars."""
        return 5000

    def _normalize_korrel8r_query(self, q: str) -> str:
        """Normalize common Korrel8r query issues for AI-provided inputs."""
        logger.info(f"Normalizing korrel8r query: {q}")
        try:
            s = (q or "").strip()
            if '\\"' in s:
                s = s.replace('\\"', '"')
            if s.startswith("alert:{"):
                s = s.replace("alert:{", "alert:alert:{", 1)

            domain = s.split(":", 1)[0] if ":" in s else ""

            if s.lower().startswith("k8s:alert:"):
                s = "alert:alert:" + s.split(":", 2)[2]
                domain = "alert"

            m = re.search(r"\{(.*)\}", s)
            if m:
                inner = m.group(1)

                if domain == "alert":
                    def repl_alert(match: re.Match) -> str:
                        key = match.group(1)
                        return f'"{key}":' + match.group(2)
                    inner2 = re.sub(r"\b([A-Za-z0-9_\.]+)\s*=\s*(\")", repl_alert, inner)
                else:
                    def repl_generic(match: re.Match) -> str:
                        key = match.group(1)
                        return f'"{key}":=' + match.group(2)
                    inner2 = re.sub(r"\b([A-Za-z0-9_\.]+)\s*=\s*(\")", repl_generic, inner)

                if inner2 != inner:
                    s = s[: m.start(1)] + inner2 + s[m.end(1) :]
            logger.info(f"Normalized korrel8r query: {s}")
            return s
        except Exception:
            return q

    def _route_tool_call_to_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Route tool call via tool executor."""
        logger.info(f"Routing tool call: {tool_name} with arguments: {arguments}")

        if tool_name == "korrel8r_get_correlated":
            try:
                q = arguments.get("query") if isinstance(arguments, dict) else None
                if isinstance(q, str) and q:
                    normalized_q = self._normalize_korrel8r_query(q)
                    if normalized_q != q:
                        logger.info(f"Normalized korrel8r query from '{q}' to '{normalized_q}'")
                        arguments = dict(arguments)
                        arguments["query"] = normalized_q
            except Exception:
                pass

        try:
            logger.info(f"Executing tool '{tool_name}' via tool executor")
            result = self.tool_executor.call_tool(tool_name, arguments)
            logger.info(f"Tool {tool_name} returned result (length: {len(result) if result else 0})")
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _inject_namespace_into_promql(self, query: str, namespace: str) -> str:
        """Inject a namespace label filter into a PromQL query string."""
        if f'namespace="' in query or f"namespace='" in query or 'namespace=~' in query or 'namespace!=' in query or 'namespace!~' in query:
            modified = re.sub(
                r'namespace\s*(?:!?=~?|!~)\s*["\'][^"\']*["\']',
                f'namespace="{namespace}"',
                query
            )
            if modified != query:
                logger.info(f"Replaced namespace in PromQL: {query} -> {modified}")
            return modified

        original = query

        def inject_ns(match):
            metric = match.group(1)
            labels = match.group(2) or ''
            rest = match.group(3) or ''

            if metric.lower() in _PROMQL_SKIP:
                return match.group(0)

            if labels:
                inner = labels[1:-1].strip()
                if inner:
                    return f'{metric}{{{inner},namespace="{namespace}"}}{rest}'
                else:
                    return f'{metric}{{namespace="{namespace}"}}{rest}'
            else:
                return f'{metric}{{namespace="{namespace}"}}{rest}'

        modified = re.sub(
            r'([a-zA-Z_:][a-zA-Z0-9_:]*)'
            r'(\{[^}]*\})?'
            r'(\[[^\]]*\])?',
            inject_ns,
            query
        )

        if modified != original:
            logger.info(f"Injected namespace '{namespace}' into PromQL: {original} -> {modified}")
        return modified

    def _get_tool_result(self, tool_name: str, tool_args: Dict[str, Any], namespace: Optional[str] = None) -> str:
        """Execute tool call with namespace injection and result truncation."""
        if namespace and tool_name in self._namespace_aware_tools:
            if tool_name == 'execute_promql' and 'query' in tool_args:
                tool_args = dict(tool_args)
                tool_args['query'] = self._inject_namespace_into_promql(
                    tool_args['query'], namespace
                )
            elif not tool_args.get('namespace'):
                tool_args = dict(tool_args)
                tool_args['namespace'] = namespace
                logger.info(f"Injected namespace '{namespace}' into {tool_name} args")

        logger.info(f"Requesting tool: {tool_name} with args: {tool_args}")

        tool_result = self._route_tool_call_to_mcp(tool_name, tool_args)

        logger.info(f"Returning result for tool {tool_name}: {str(tool_result)[:200]}...")

        max_length = self._get_max_tool_result_length()
        if isinstance(tool_result, str) and len(tool_result) > max_length:
            logger.info(f"Truncating result from {len(tool_result)} to {max_length} chars")
            tool_result = tool_result[:max_length] + "\n... [Result truncated due to size]"
        else:
            logger.info(f"Tool result size: {len(str(tool_result))} chars (within limit of {max_length})")

        return tool_result

    @abstractmethod
    def chat(
        self,
        user_question: str,
        namespace: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Chat with the model. Must be implemented by subclasses."""
