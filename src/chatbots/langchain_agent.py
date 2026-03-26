"""
LangGraph Agent - Unified LangChain/LangGraph replacement for provider-specific bots.

This module implements a LangGraphAgent that replaces the existing 4 bot classes
(openai_bot, anthropic_bot, google_bot, llama_bot) with a single LangGraph StateGraph
backed by LangChain ChatModels.
"""

import json
import re
from typing import Annotated, Optional, List, Dict, Any, Callable, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chatbots.tool_executor import ToolExecutor
from common.pylogger import get_python_logger
from core.config import NAMESPACE_AWARE_TOOLS

logger = get_python_logger()

# ---------------------------------------------------------------------------
# Utility imports from base.py (standalone functions)
# ---------------------------------------------------------------------------
from chatbots.base import (
    _PROMQL_SKIP,
)


# ---------------------------------------------------------------------------
# Standalone utility functions (mirrored from base.py class methods so that
# the agent can use them without inheriting from BaseChatBot)
# ---------------------------------------------------------------------------

def inject_namespace_into_promql(query: str, namespace: str) -> str:
    """Inject a namespace label filter into a PromQL query string.

    Handles common PromQL patterns:
    - metric_name -> metric_name{namespace="ns"}
    - metric_name{existing="label"} -> metric_name{existing="label",namespace="ns"}
    - sum(metric_name) -> sum(metric_name{namespace="ns"})
    - sum(rate(metric_name[5m])) -> sum(rate(metric_name{namespace="ns"}[5m]))

    If the query already contains a namespace filter, it is replaced with
    the active namespace to ensure the UI dropdown selection takes precedence.
    """
    if (
        f'namespace="' in query
        or f"namespace='" in query
        or 'namespace=~' in query
        or 'namespace!=' in query
        or 'namespace!~' in query
    ):
        modified = re.sub(
            r'namespace\s*(?:!?=~?|!~)\s*["\'][^"\']*["\']',
            f'namespace="{namespace}"',
            query,
        )
        if modified != query:
            logger.info("Replaced namespace in PromQL: %s -> %s", query, modified)
        return modified

    original = query

    def _inject_ns(match: re.Match) -> str:
        metric = match.group(1)
        labels = match.group(2) or ""
        rest = match.group(3) or ""

        if metric.lower() in _PROMQL_SKIP:
            return match.group(0)

        if labels:
            inner = labels[1:-1].strip()
            if inner:
                return f'{metric}{{{inner},namespace="{namespace}"}}{rest}'
            return f'{metric}{{namespace="{namespace}"}}{rest}'
        return f'{metric}{{namespace="{namespace}"}}{rest}'

    modified = re.sub(
        r'([a-zA-Z_:][a-zA-Z0-9_:]*)'
        r'(\{[^}]*\})?'
        r'(\[[^\]]*\])?',
        _inject_ns,
        query,
    )

    if modified != original:
        logger.info("Injected namespace '%s' into PromQL: %s -> %s", namespace, original, modified)
    return modified


def normalize_korrel8r_query(q: str) -> str:
    """Normalize common Korrel8r query issues for AI-provided inputs.

    - Ensure domain:class present for alert domain (alert -> alert:alert)
    - Convert selector keys of form key="value" to JSON-style "key":="value"
    """
    logger.info("Normalizing korrel8r query: %s", q)
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
                def _repl_alert(match: re.Match) -> str:
                    key = match.group(1)
                    return f'"{key}":' + match.group(2)
                inner2 = re.sub(r"\b([A-Za-z0-9_\.]+)\s*=\s*(\")", _repl_alert, inner)
            else:
                def _repl_generic(match: re.Match) -> str:
                    key = match.group(1)
                    return f'"{key}":=' + match.group(2)
                inner2 = re.sub(r"\b([A-Za-z0-9_\.]+)\s*=\s*(\")", _repl_generic, inner)

            if inner2 != inner:
                s = s[: m.start(1)] + inner2 + s[m.end(1) :]
        logger.info("Normalized korrel8r query: %s", s)
        return s
    except Exception:
        return q


def get_tool_result(
    tool_executor: ToolExecutor,
    tool_name: str,
    tool_args: Dict[str, Any],
    namespace: Optional[str],
    namespace_aware_tools: frozenset,
    max_length: int,
) -> str:
    """Execute a tool call via the ToolExecutor with namespace injection, korrel8r
    normalization, and result truncation.

    Args:
        tool_executor: The ToolExecutor instance for calling tools.
        tool_name: Name of the tool to call.
        tool_args: Arguments to pass to the tool.
        namespace: Optional namespace to inject into tool args.
        namespace_aware_tools: Set of tool names that accept namespace.
        max_length: Maximum character length before truncation.

    Returns:
        Tool result string (possibly truncated).
    """
    # Inject namespace for namespace-scoped queries
    if namespace and tool_name in namespace_aware_tools:
        if tool_name == "execute_promql" and "query" in tool_args:
            tool_args = dict(tool_args)
            tool_args["query"] = inject_namespace_into_promql(tool_args["query"], namespace)
        elif not tool_args.get("namespace"):
            tool_args = dict(tool_args)
            tool_args["namespace"] = namespace
            logger.info("Injected namespace '%s' into %s args", namespace, tool_name)

    # Normalize Korrel8r query inputs
    if tool_name == "korrel8r_get_correlated":
        q = tool_args.get("query") if isinstance(tool_args, dict) else None
        if isinstance(q, str) and q:
            normalized_q = normalize_korrel8r_query(q)
            if normalized_q != q:
                logger.info("Normalized korrel8r query from '%s' to '%s'", q, normalized_q)
                tool_args = dict(tool_args)
                tool_args["query"] = normalized_q

    logger.info("Requesting tool: %s with args: %s", tool_name, tool_args)

    try:
        result = tool_executor.call_tool(tool_name, tool_args)
        logger.info("Tool %s returned result (length: %d)", tool_name, len(result) if result else 0)
    except Exception as e:
        logger.error("Error calling tool %s: %s", tool_name, e)
        return f"Error executing {tool_name}: {str(e)}"

    logger.info("Returning result for tool %s: %s...", tool_name, str(result)[:200])

    if isinstance(result, str) and len(result) > max_length:
        logger.info("Truncating result from %d to %d chars", len(result), max_length)
        result = result[:max_length] + "\n... [Result truncated due to size]"
    else:
        logger.info("Tool result size: %d chars (within limit of %d)", len(str(result)), max_length)

    return result


def format_scope_line(namespace: Optional[str]) -> str:
    """Format the Scope line for the system prompt."""
    if namespace:
        return (
            "**NAMESPACE-SCOPED: %s** -- ALL queries MUST be filtered "
            "to this namespace only" % namespace
        )
    return "Cluster-wide analysis"


def format_namespace_directive(namespace: Optional[str]) -> str:
    """Format the namespace scoping directive for the system prompt."""
    if not namespace:
        return ""
    return (
        "\n**NAMESPACE SCOPE REQUIREMENT -- MANDATORY:**\n"
        'You are operating in NAMESPACE-SCOPED mode for namespace **"%s"**.\n'
        "- You MUST call tools (execute_promql, search_metrics, etc.) to get FRESH data for this namespace.\n"
        "- Do NOT reuse or reference results from previous queries -- they may be from a different scope.\n"
        '- Do NOT add namespace filters to your PromQL queries -- the system automatically injects namespace="%s" for you.\n'
        '- If a previous conversation shows cluster-wide data, IGNORE it and query fresh data for "%s".\n'
        '- Every answer must reflect data from namespace "%s" ONLY.\n'
        '- If the user mentions a different namespace in their question, still answer the question but use "%s"'
        " -- the active namespace selected in the UI always takes precedence.\n"
    ) % (namespace, namespace, namespace, namespace, namespace)


def get_base_prompt(namespace: Optional[str] = None) -> str:
    """Create the base system prompt shared by all models."""
    prompt = f"""You are an expert Kubernetes and Prometheus observability assistant.

**PRIMARY RULE: ANSWER ONLY WHAT THE USER ASKS. DO NOT EXPLORE BEYOND THEIR SPECIFIC QUESTION.**

You have access to monitoring tools and should provide focused, targeted responses.

**Your Environment:**
- Cluster: OpenShift with AI/ML workloads, GPUs, and comprehensive monitoring
- Scope: {format_scope_line(namespace)}
- Tools: Direct access to Prometheus/Thanos metrics via MCP tools
{format_namespace_directive(namespace)}

**Available Tools:**

**Core Observability Tools:**
- search_metrics: Pattern-based metric search - use for broad exploration
- execute_promql: Execute PromQL queries for actual data
- convert_time_to_promql_duration: Convert decimal hours to Prometheus format (use before constructing queries with decimal time values)
- get_metric_metadata: Get detailed information about specific metrics
- get_label_values: Get available label values
- suggest_queries: Get PromQL suggestions based on user intent
- explain_results: Get human-readable explanation of query results
- **get_metrics_categories**: Get all metric categories with summary (NEW - use for exploring available metrics by category)
- **search_metrics_by_category**: Search metrics filtered by category and priority (NEW - use for targeted category-specific queries)

**Trace Analysis Tools:**
- chat_tempo_tool: Conversational trace analysis - use for trace/span/latency/request flow questions
- query_tempo_tool: Direct tempo queries for specific trace searches
- get_trace_details_tool: Get detailed information about specific trace IDs

**Log Analysis Tools:**
- get_correlated_logs: Fetch application and infrastructure logs for a namespace or pod. Use for: "show me logs", "error logs in namespace X", "what's happening in pod Y", crash investigation, log search by severity. Requires namespace; pod is optional.

**Correlation & Advanced Analysis:**
- korrel8r_get_correlated: Get correlated observability data across domains (find logs/traces/metrics related to alerts) - available if Korrel8r is configured. Preferred over korrel8r_query_objects (for investigation and correlation).
- korrel8r_query_objects: Query for specific observability objects (alerts, logs, traces, metrics) - available if Korrel8r is configured. Use for direct data access only.

**Note:** Additional specialized tools are available for specific use cases (VLLM metrics, OpenShift analysis, model management, etc.) and will be provided to you automatically via the function calling interface when needed.

**Tool Selection Guidelines:**

**For Trace-Related Questions (trace, span, latency, request flow, distributed tracing, performance):**
- Use `chat_tempo_tool` for conversational trace analysis with natural language questions
- Use `query_tempo_tool` for specific trace queries when you need targeted data
- Use `get_trace_details_tool` for detailed analysis of specific trace IDs
- Extract time ranges from natural language ("last 24 hours", "yesterday", "last week")

**For Alert Queries:**

**Smart Two-Phase Approach:**
- Start with Prometheus (fast, simple) for basic alert data
- Escalate to Korrel8r only when needed for correlation or explicitly requested

**1. USER EXPLICITLY REQUESTS KORREL8r ("use korrel8r", "query korrel8r")**:
   - ALWAYS use Korrel8r tools immediately (korrel8r_query_objects or korrel8r_get_correlated)
   - Query format: `alert:alert:{{"alertname":"AlertName"}}`
   - Examples: "Use korrel8r to investigate AlertExampleDown", "Query korrel8r for HighCPU alert"

**2. USER ASKS FOR INVESTIGATION/CORRELATION** (without mentioning korrel8r):
   - Phase 1: Use `execute_promql` with ALERTS metric to get alert details
   - Phase 2: Use Korrel8r to find related logs/traces/metrics
   - Examples: "Investigate AlertExampleDown", "What's related to HighCPU alert?", "Find correlated data for alert X"

**3. BASIC ALERT QUERIES** (listing/checking status only):
   - Use ONLY `execute_promql` with the `ALERTS` metric - DO NOT use Korrel8r
   - Query firing alerts: `ALERTS{{alertstate="firing"}}`
   - Query specific alerts: `ALERTS{{alertstate="firing", alertname="HighCPU"}}`
   - Examples: "Any alerts firing?", "Show me alerts", "List all critical alerts", "Check alert status"

**For Log Queries (logs, errors, pod output, crash investigation, "what happened"):**
- Use `get_correlated_logs` for namespace/pod log retrieval -- pass the namespace (required) and optionally a pod name
- Use `korrel8r_get_correlated` when you need to correlate logs with traces and metrics (cross-signal investigation)
- Examples: "Show me logs for namespace llm-serving", "Error logs from pod vllm-predictor", "What happened in namespace gpu-workloads?", "Show me crash logs"

**Your Intelligence Style:**
1. **Rich Contextual Analysis**: Provide context, thresholds, and implications -- not just raw numbers
2. **Intelligent Grouping**: Group related pods by function (AI/ML Stack, Infrastructure, Data Storage) with counts
3. **Operational Intelligence**: Include health assessments, trend context, and actionable recommendations

**CORE PRINCIPLES:**
- Use tools to get real data -- NEVER fabricate numbers or metric names
- Be thorough but focused -- answer what was asked, nothing more
- STOP when you have enough data to answer the question well

**vLLM / Model-Serving Metrics:**
Use `search_metrics_by_category` with category `gpu_ai` to discover exact metric names.
Key metrics (DO NOT guess names -- always search first):
- Latency: `vllm:e2e_request_latency_seconds`, `vllm:time_to_first_token_seconds`
- Throughput: `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:num_requests_total`
- Cache: `vllm:gpu_cache_usage_perc`, `vllm:cpu_cache_usage_perc`
- GPU: `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_UTIL`
For decimal hour time ranges (e.g., "2.3 hours"), use `convert_time_to_promql_duration()` to get the correct PromQL format.

**Tool Selection Rules (ALWAYS follow these):**
- Traces/spans/latency -> `chat_tempo_tool` (search) or `get_trace_details_tool` (by ID)
- Metrics/pods/GPU/CPU/memory -> **ALWAYS** call `search_metrics` or `search_metrics_by_category` FIRST to discover exact metric names, THEN call `execute_promql` with the discovered names
- Alerts -> `execute_promql` with `ALERTS{{alertstate="firing"}}` metric
- Logs/errors/pod output -> `get_correlated_logs` with namespace and optional pod
- Correlation/investigation/korrel8r -> `korrel8r_get_correlated` with goals and k8s query
- Pod health/failures -> `execute_promql` with `kube_pod_container_status_waiting_reason` and `kube_pod_container_status_terminated_reason`
- When the user explicitly names a tool (e.g., "use korrel8r"), ALWAYS use that tool

**Your Workflow:**
1. **Determine** what the user is asking for (trace, metrics, logs, or alerts?)
2. **Discover** -- for metrics questions, ALWAYS call `search_metrics` or `search_metrics_by_category` first
3. **Execute** the query tool with the discovered metric names
4. **Answer** with the specific data -- DONE!

**CRITICAL: Interpreting Metrics Correctly**
- **Boolean/Status Metrics**: These use VALUE to indicate state where 1 means TRUE and 0 means FALSE
  - Always check the metric VALUE not just the labels
  - Filter for value equals 1 to get actual active states
- **Gauge Metrics**: Report current state or value at a point in time
- **Counter Metrics**: Always increasing, use rate function for meaningful analysis

**CRITICAL: Always Group Metrics for Detailed Breakdowns**
- **Always use grouping by pod and namespace** for resource metrics like CPU memory GPU
- **Show detailed breakdowns** not just summary totals
- List top consumers by pod and namespace with actual names
- Categorize by workload type such as AI/ML versus Infrastructure

**Pod Health & Failure Detection:**
- `kube_pod_status_phase` only tracks pod-level phases -- most failures (CrashLoopBackOff, ImagePullBackOff, OOMKilled) are NOT in "Failed" phase
- Check container-level metrics: `kube_pod_container_status_waiting_reason` and `kube_pod_container_status_terminated_reason`
- ALWAYS append `== 1` to kube-state-metrics status queries to exclude stale time series

**PromQL Pod/Container Name Matching:**
- When querying by pod name, always use regex matching (e.g., `pod=~"name.*"`) instead of exact match (`pod="name"`), because Kubernetes pod names include deployment and replicaset hash suffixes (e.g., `my-app-6d5f8b7c4-x9k2m`).
- Apply the same regex pattern for container and deployment names that may have generated suffixes.

**Response Format:**
```
[Emoji + Summary Title]
[Key Numbers & Summary]

[Rich contextual analysis with operational insights]

**Technical Details:**
- **PromQL Used:** `your_query_here`
- **Metric Source:** metric_name_here
- **Data Points:** X samples over Y timeframe
```

**Critical Rules:**
- ALWAYS include the PromQL query in technical details
- ALWAYS use tools to get real data - never make up numbers
- ALWAYS use EXACT metric names from tool results - never modify or "normalize" metric names
- When reporting "Metric Source", copy the exact name returned by the tool
- Provide operational context and health assessments
- Use emojis and categorization for clarity
- Make responses informative and actionable
- Show conversational tool usage: "Let me check..." "I'll also look at..."

Begin by finding the perfect metric for the user's question, then provide comprehensive analysis."""

    return prompt


def create_system_prompt(
    namespace: Optional[str] = None,
    model_specific_instructions: str = "",
) -> str:
    """Create system prompt combining base prompt with optional model-specific instructions."""
    base = get_base_prompt(namespace)
    if model_specific_instructions:
        return f"{base}\n\n{model_specific_instructions}"
    return base


# ---------------------------------------------------------------------------
# Llama nudge system
# ---------------------------------------------------------------------------

# Query category patterns -> specific tool + usage hint for the nudge message.
# Order matters: first match wins. More specific patterns come first.
_QUERY_CATEGORIES = [
    # Correlation / korrel8r -- must precede "alert" because pod names
    # like "alert-example-5d9cbf68fd-62zsb" would otherwise match the
    # alert pattern and route to execute_promql instead.
    (
        re.compile(r"correlat|korrel8r|investigate", re.IGNORECASE),
        "korrel8r_get_correlated",
        'Call korrel8r_get_correlated with goals=["trace:span","log:application","log:infrastructure"] '
        'and query="k8s:Pod:{\\"namespace\\":\\"<namespace>\\",\\"name\\":\\"<pod>\\"}"',
    ),
    # Alerts -- only match "alert" as a standalone concept (firing alerts),
    # not as part of pod/deployment names like "alert-example".
    (
        re.compile(r"\balerts?\b(?!-)", re.IGNORECASE),
        "execute_promql",
        'Call execute_promql with query: ALERTS{namespace="<namespace>"} or ALERTS{} for cluster-wide.',
    ),
    # Pod health / failures
    (
        re.compile(r"pod.*(fail|crash|error|unhealthy|status|restart)", re.IGNORECASE),
        "execute_promql",
        "Call execute_promql with query: "
        'kube_pod_container_status_waiting_reason{namespace="<namespace>", '
        'reason=~"CrashLoopBackOff|ImagePullBackOff|ErrImagePull"} == 1',
    ),
    # Trace details (specific trace ID)
    (
        re.compile(r"trace.*(detail|id|info)|detail.*trace", re.IGNORECASE),
        "get_trace_details_tool",
        "Call get_trace_details_tool with the trace ID from the user query.",
    ),
    # Traces (general)
    (
        re.compile(r"trace|span|latency", re.IGNORECASE),
        "chat_tempo_tool",
        "Call chat_tempo_tool with the user query to search for traces.",
    ),
    # Logs
    (
        re.compile(r"\blog", re.IGNORECASE),
        "get_correlated_logs",
        "Call get_correlated_logs with namespace and optional pod name.",
    ),
    # GPU metrics
    (
        re.compile(r"gpu|power|temperature|temp\b", re.IGNORECASE),
        "execute_promql",
        "Call execute_promql for each metric separately. "
        "GPU power: avg(DCGM_FI_DEV_POWER_USAGE) by (pod, namespace). "
        "GPU temperature: avg(DCGM_FI_DEV_GPU_TEMP) by (pod, namespace).",
    ),
    # CPU / memory metrics
    (
        re.compile(r"cpu|memory|mem\b", re.IGNORECASE),
        "execute_promql",
        "Call execute_promql. "
        "CPU: sum(rate(container_cpu_usage_seconds_total[5m])) by (pod, namespace). "
        "Memory: sum(container_memory_usage_bytes) by (pod, namespace).",
    ),
    # Generic metrics
    (
        re.compile(r"metric|promql|prometheus", re.IGNORECASE),
        "execute_promql",
        "Call execute_promql with the appropriate PromQL query.",
    ),
]


def _get_nudge_for_query(
    user_question: str, namespace: Optional[str] = None
) -> tuple:
    """Build a category-specific nudge message based on the user's query.

    Returns (nudge_text, tool_name) where tool_name is the matched tool
    or None if no category matched.
    """
    for pattern, tool_name, hint in _QUERY_CATEGORIES:
        if pattern.search(user_question):
            resolved_hint = hint
            if namespace:
                resolved_hint = resolved_hint.replace("<namespace>", namespace)
            else:
                resolved_hint = re.sub(r'namespace="<namespace>",?\s*', "", resolved_hint)
                resolved_hint = resolved_hint.replace("<namespace>", "")
            logger.info("Smart nudge matched category: tool=%s", tool_name)
            nudge_text = (
                f"You MUST call the `{tool_name}` tool now to answer this question. "
                f"Do NOT fabricate data. {resolved_hint}"
            )
            return nudge_text, tool_name

    # Generic fallback nudge
    return (
        "You MUST use the provided tools to answer this question. "
        "Do NOT fabricate data. Call the appropriate tool now.",
        None,
    )


# ---------------------------------------------------------------------------
# LangGraph StateGraph Agent
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State schema for the LangGraph agent."""
    messages: Annotated[list, add_messages]


class LangGraphAgent:
    """LangGraph-based agent that unifies all provider-specific chat bot logic.

    Uses a LangChain ChatModel (ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI,
    etc.) with LangGraph's StateGraph for the agentic tool-calling loop.

    Args:
        chat_model: A LangChain ChatModel instance bound with tools.
        tool_executor: The existing ToolExecutor instance (for namespace injection,
            korrel8r normalization, and actual tool execution).
        tools: List of LangChain tools (from the tools.py bridge).
        model_name: The original model name string for logging/identification.
        is_local_llama: Whether this is a local Llama model (enables nudge system).
        max_tool_result_length: Per-model truncation limit for tool results.
    """

    def __init__(
        self,
        chat_model,
        tool_executor: ToolExecutor,
        tools: list,
        model_name: str,
        is_local_llama: bool = False,
        max_tool_result_length: int = 10000,
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

        self.chat_model = chat_model
        self.tool_executor = tool_executor
        self.tools = tools
        self.model_name = model_name
        self.is_local_llama = is_local_llama
        self.max_tool_result_length = max_tool_result_length
        self._namespace_aware_tools = NAMESPACE_AWARE_TOOLS

        # Build a lookup of tool names for convenience
        self._tool_names = {t.name for t in tools}

        # Bind tools to the chat model so that the LLM knows about them
        self._model_with_tools = chat_model.bind_tools(tools)

        # Build the graph
        self._graph = self._build_graph()

        logger.info(
            "LangGraphAgent initialized with model: %s (is_local_llama=%s, "
            "max_tool_result_length=%d, tools=%d)",
            self.model_name,
            self.is_local_llama,
            self.max_tool_result_length,
            len(self.tools),
        )

    # ----- graph construction ------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        graph.add_node("call_model", self._call_model_node)
        graph.add_node("execute_tools", self._execute_tools_node)

        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model",
            self._should_continue,
            {"execute_tools": "execute_tools", END: END},
        )
        graph.add_edge("execute_tools", "call_model")

        return graph.compile()

    # ----- graph nodes -------------------------------------------------------

    def _call_model_node(self, state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        messages = state["messages"]
        response = self._model_with_tools.invoke(messages)
        return {"messages": [response]}

    def _execute_tools_node(self, state: AgentState) -> dict:
        """Execute tool calls from the last AI message.

        Routes through the ToolExecutor with namespace injection,
        korrel8r normalization, and result truncation -- rather than calling
        LangChain tools directly.
        """
        last_message: AIMessage = state["messages"][-1]
        namespace = state.get("namespace")  # type: ignore[typeddict-item]
        progress_callback = state.get("progress_callback")  # type: ignore[typeddict-item]

        tool_messages: list[ToolMessage] = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call["id"]

            if progress_callback:
                progress_callback(f"Using tool: {tool_name}")

            result = get_tool_result(
                tool_executor=self.tool_executor,
                tool_name=tool_name,
                tool_args=tool_args,
                namespace=namespace,
                namespace_aware_tools=self._namespace_aware_tools,
                max_length=self.max_tool_result_length,
            )

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call_id)
            )

        return {"messages": tool_messages}

    # ----- conditional edge --------------------------------------------------

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        """Decide whether to execute tools or end the graph."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "execute_tools"
        return END

    # ----- public API --------------------------------------------------------

    def chat(
        self,
        user_question: str,
        namespace: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Chat with the model, matching the existing bot interface.

        Args:
            user_question: The user's question.
            namespace: Optional namespace filter for scoping queries.
            progress_callback: Optional callback for progress updates.
            conversation_history: Optional list of previous messages in format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            The model's final response as a string.
        """
        logger.info(
            "LangGraphAgent.chat() - model: %s, namespace: %s",
            self.model_name,
            namespace,
        )

        try:
            # Build system prompt
            system_prompt = create_system_prompt(namespace)

            # Construct initial message list
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

            # Append conversation history
            if conversation_history:
                logger.info(
                    "Adding %d messages from conversation history",
                    len(conversation_history),
                )
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

            # Append the current user question
            messages.append(HumanMessage(content=user_question))

            if progress_callback:
                progress_callback("Thinking...")

            # Prepare the initial state.  namespace and progress_callback are
            # passed as extra keys on the state dict so that execute_tools can
            # access them.  They are not part of the formal AgentState TypedDict
            # (which only declares ``messages``) but LangGraph forwards all keys.
            initial_state = {
                "messages": messages,
                "namespace": namespace,
                "progress_callback": progress_callback,
            }

            config = {"recursion_limit": 30}

            # -----------------------------------------------------------
            # Llama nudge system: invoke the graph once.  If the model
            # returned without calling tools (possible fabrication), add
            # a nudge message and re-invoke with the extended history.
            # -----------------------------------------------------------
            result_state = self._graph.invoke(initial_state, config=config)

            if self.is_local_llama:
                result_state = self._apply_llama_nudge_if_needed(
                    result_state, user_question, namespace, progress_callback, config
                )

            # Extract the final AI message content.
            # Google Gemini may return content as a list of content blocks
            # rather than a plain string, so normalise to str.
            final_message = result_state["messages"][-1]
            raw_content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )
            if isinstance(raw_content, list):
                final_content = "\n".join(
                    block.get("text", str(block))
                    if isinstance(block, dict)
                    else str(block)
                    for block in raw_content
                )
            else:
                final_content = raw_content

            if not final_content or not final_content.strip():
                logger.warning(
                    "LangGraphAgent produced empty response for model %s",
                    self.model_name,
                )
                return (
                    "I wasn't able to retrieve data for this query. "
                    "Please try rephrasing your question or being more "
                    "specific about what information you need."
                )

            logger.info(
                "LangGraphAgent.chat() completed for model %s (response length: %d)",
                self.model_name,
                len(final_content),
            )
            return final_content

        except Exception as e:
            logger.error("Error in LangGraphAgent.chat(): %s", e)
            import traceback
            logger.error(traceback.format_exc())
            return f"Error during LangGraph agent execution: {str(e)}"

    # ----- Llama nudge helpers -----------------------------------------------

    def _apply_llama_nudge_if_needed(
        self,
        result_state: dict,
        user_question: str,
        namespace: Optional[str],
        progress_callback: Optional[Callable],
        config: dict,
    ) -> dict:
        """Check whether the Llama model fabricated an answer (returned text
        without calling any tools on the first pass).  If so, append a nudge
        message and re-invoke the graph.

        Returns the (possibly updated) result state.
        """
        # Walk through the messages to see if any tool calls were made
        has_tool_calls = any(
            hasattr(m, "tool_calls") and m.tool_calls
            for m in result_state["messages"]
        )

        if has_tool_calls:
            # Model used tools -- no nudge needed
            return result_state

        logger.warning(
            "Llama returned text without calling any tools -- possible fabrication. "
            "Nudging model with category-specific hint."
        )

        nudge_text, _matched_tool = _get_nudge_for_query(user_question, namespace)

        # Build an extended message list from the current result, then append
        # the nudge as a new human message.
        extended_messages = list(result_state["messages"])
        extended_messages.append(HumanMessage(content=nudge_text))

        nudge_state = {
            "messages": extended_messages,
            "namespace": namespace,
            "progress_callback": progress_callback,
        }

        if progress_callback:
            progress_callback("Retrying with tool guidance...")

        nudge_result = self._graph.invoke(nudge_state, config=config)
        return nudge_result
