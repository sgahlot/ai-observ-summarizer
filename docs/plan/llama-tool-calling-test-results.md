# Llama-3.1-8B Tool Calling Test Results

**Branch:** `appeng-4577-reduce_system_prompt_bloat`
**Model:** `meta-llama/Llama-3.1-8B-Instruct` (served from `main` namespace)
**Test setup:** Local dev (`./scripts/local-dev.sh`) with MCP server and React UI running locally
**Related:** [System Prompt Bloat Fix](system-prompt-bloat.md) | [LLM Tool Calling Fix](llm-tool-calling-fix.md)

---

## Test Queries

These queries cover the primary use cases: GPU metrics, alerts, pod investigation, pod failures, traces, and correlation. Use the same queries across all models for comparison.

| # | Query | Scope | Category |
|---|-------|-------|----------|
| Q1 | "Show me GPU power consumption and temperature trends" | Cluster-wide | GPU metrics |
| Q2 | "Any alerts firing in jianrong namespace" | Cluster-wide | Alerts |
| Q3 | "Use correlated data to investigate pod my-app-example-wgvun in jianrong namespace" | Cluster-wide | Korrel8r / correlation |
| Q4 | "Use korrel8r to investigate pod my-app-example-wgvun" | Namespace-scoped (jianrong) | Korrel8r / correlation |
| Q5 | "any pods failing in openshift-monitoring namespace" | Cluster-wide | Pod health |
| Q6 | "any pods failing" | Namespace-scoped (jianrong) | Pod health |
| Q7 | "any pods failing in jianrong namespace" | Cluster-wide | Pod health |
| Q8 | "Find the top trace and find its details" (follow-up) | Cluster-wide | Traces |
| Q9 | "Give me trace details for trace id 1234567890abcdef" | Cluster-wide | Traces |
| Q10 | "Give me trace details for trace id 25e4d24e123515447ac0fbe8e64bf179" | Cluster-wide | Traces |

---

## Evaluation Criteria

For each query, record:

| Criterion | What to check |
|-----------|--------------|
| **Tools called?** | Did the model invoke tools via the function calling API? Check MCP server log for "Using tool:" entries |
| **Correct tool?** | Did it pick the right tool for the query? (e.g., `execute_promql` for metrics, `korrel8r_get_correlated` for correlation) |
| **Real data?** | Is the response based on actual tool results, not fabricated numbers? |
| **Response quality** | Is the data summarized with context, or raw JSON dumped? |
| **Nudge fired?** | Did the fabrication guard or text-tool-call guard trigger? Check log for "Nudging model" |
| **Error?** | Any context overflow (400), timeout, or other errors? |

---

## Results: Llama-3.1-8B (Before Fix)

**Date:** 2026-03-03
**Code:** `main` branch code (no nudge guards, no tool allowlist, full system prompt)
**Score: 1 / 10 queries succeeded**

| # | Query | Tools Called? | Failure Mode | Notes |
|---|-------|-------------|--------------|-------|
| Q1 | GPU power & temperature | No | **Fabrication** | Fake data: 250W, 350W, 60C, 80C. Round numbers, invented pod names |
| Q2 | Alerts firing in jianrong | No | **Fabrication** | Invented alert names and details |
| Q3 | Investigate pod (correlated) | No | **Fabrication** | Invented pod names |
| Q4 | Investigate pod (korrel8r) | No | **Raw JSON as text** | Output `{"type":"function","name":"korrel8r_query_objects",...}` as prose |
| Q5 | Pods failing (openshift-monitoring) | No | **Fabrication** | Fake data |
| Q6 | Pods failing (namespace-scoped) | No | **Raw JSON as text** | Output raw JSON tool call as text |
| Q7 | Pods failing (jianrong, cluster-wide) | No | **Fabrication** | Fake data |
| Q8 | Find top trace (follow-up) | No | **Fabrication** | Made-up trace ID `trace-1234567890` |
| Q9 | Trace details (fake ID) | No | **Fabrication** | 12 hallucinated spans with realistic names |
| Q10 | Trace details (real ID) | **Yes** | **Success** | Called `get_trace_details_tool` correctly |

### Failure mode summary (before fix)

| Failure Mode | Count | Queries |
|-------------|-------|---------|
| Fabrication (fake data, no tools called) | 7 | Q1, Q2, Q3, Q5, Q7, Q8, Q9 |
| Raw JSON as text (tool call written as prose) | 2 | Q4, Q6 |
| Success | 1 | Q10 |

### Root causes identified

1. **No fabrication guard:** `llama_bot.py` silently returned `message.content` when model returned `finish_reason=stop` without calling tools
2. **No text-tool-call detection:** Raw JSON tool calls in text were passed through as the final response
3. **42 tool definitions:** Combined tool schemas consumed ~9,200 tokens (65% of 14,336 token limit)
4. **Large system prompt:** ~10,600 chars of system prompt + model-specific instructions left minimal room for tool results

---

## Results: Llama-3.1-8B (After Fix)

**Date:** 2026-03-03
**Code:** `appeng-4577-reduce_system_prompt_bloat` branch
**Fixes applied:** Fabrication guard, text-tool-call detection, `tool_choice="required"` on retry, tool allowlist (42 -> 19), compact system prompt

| # | Query | Tools Called? | Nudge Fired? | Response Quality | Notes |
|---|-------|-------------|-------------|-----------------|-------|
| Q1 | GPU power & temperature | **Yes** (`execute_promql`) | Yes (fabrication guard) | Poor — raw JSON dump, only queried power not temperature | Tool called but response needs improvement |
| Q2 | Alerts firing in jianrong | | | | |
| Q3 | Investigate pod (correlated) | | | | |
| Q4 | Investigate pod (korrel8r) | | | | |
| Q5 | Pods failing (openshift-monitoring) | | | | |
| Q6 | Pods failing (namespace-scoped) | | | | |
| Q7 | Pods failing (jianrong, cluster-wide) | | | | |
| Q8 | Find top trace (follow-up) | | | | |
| Q9 | Trace details (fake ID) | | | | |
| Q10 | Trace details (real ID) | | | | |

### Q1 detailed observations

- **First run:** Fabrication guard fired on iteration 1. Nudge retry with `tool_choice="required"` sent, but LlamaStack did not enforce it — model returned stop again without calling tools. Response was empty/near-empty.
- **Second run (same query):** Model called `execute_promql` on iteration 1 (no nudge needed). Tool returned 19,366 chars, truncated to 8,000. But iteration 2 hit **400 Bad Request: context length 14,336 exceeded (14,525 tokens)**. This was caused by 42 tool definitions + large system prompt + tool result.
- **After tool allowlist + compact prompt:** Model called `execute_promql` successfully. No context overflow. But response quality was poor:
  - Only queried power, not temperature (user asked for both)
  - Dumped raw JSON instead of summarizing
  - Confused power and temperature data (same series described as both)

---

## Results: OpenAI (Baseline Comparison)

**Date:** 2026-03-03

| # | Query | Tools Called? | Response Quality | Notes |
|---|-------|-------------|-----------------|-------|
| Q10 | Trace details (real ID) | **Yes** | Good | Properly formatted, summarized trace spans |

---

## Results: Other Models

_To be filled in as tests are run with other models (Gemini, Claude, etc.) for comparison._

### Gemini

| # | Query | Tools Called? | Response Quality | Notes |
|---|-------|-------------|-----------------|-------|
| | | | | |

### Claude

| # | Query | Tools Called? | Response Quality | Notes |
|---|-------|-------------|-----------------|-------|
| | | | | |

---

## Fix Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2026-03-03 | Add fabrication guard (iteration 1, no tools -> nudge) | Catches 7/10 failure cases |
| 2026-03-03 | Add `_detect_text_tool_calls()` to LlamaChatBot | Catches 2/10 failure cases (raw JSON as text) |
| 2026-03-03 | Add `tool_choice="required"` on nudge retry | Forces API-level tool call on retry |
| 2026-03-03 | Add tool allowlist (42 -> 19 tools) | Saves ~5,000 tokens of context |
| 2026-03-03 | Override `_get_base_prompt()` with compact version | Saves ~2,700 tokens (10,600 -> ~1,600 chars) |
| 2026-03-03 | Add `_get_tool_allowlist()` hook to BaseChatBot | Enables per-model tool filtering |
