# System Prompt Bloat After dev Merge into add-loki-log-support

**Branch:** `appeng-4577-reduce_system_prompt_bloat`
**Status:** Part 2 implemented — see Completed Actions below
**Last Updated:** 2026-03-03
**Related:** [LLM Tool Calling Degradation Fix](llm-tool-calling-fix.md) (Part 2) | [Llama Tool Calling Test Results](llama-tool-calling-test-results.md)

---

## Context

The `feature` branch (which `add-loki-log-support` was based on) was merged into `dev`. Merging `dev` back into `add-loki-log-support` brought in all the accumulated changes from the `feature` branch, including the system prompt additions from PR #218 (Smart Metrics Catalog) and the namespace support work.

The [LLM Tool Calling Fix plan](llm-tool-calling-fix.md) documents that the system prompt grew from ~7,400 to ~11,470 characters (54% increase) after PR #218, which caused Gemini to deprioritize function calling and fall back to text-based tool descriptions. Part 1 (runtime mitigations) was completed; Part 2 (reducing the prompt) was planned but not yet applied.

The `dev` merge brought in the bloated prompt that Part 2 was supposed to trim.

---

## What the Merge Added to `_get_base_prompt()` in `base.py`

| Addition | Approx. Size | Risk Level | Notes |
|----------|-------------|------------|-------|
| "Enhanced Metrics Catalog" environment line | ~80 chars | Low | Useful context for the model |
| Catalog tools (`get_metrics_categories`, `search_metrics_by_category`) | ~250 chars | Low | Necessary tool guidance |
| **Duplicate "Trace Analysis Tools" block** | ~250 chars | **Bug** | Identical 3-tool section appears twice (lines 554-563) |
| `get_correlated_logs` tool section | ~350 chars | Low | New tool from this branch, needed |
| **vLLM / Model-Serving Domain Knowledge block** | ~2,200 chars | **High** | Largest addition; lists every vLLM metric with PromQL examples inline |
| **"Enhanced Metrics Catalog" footer block** | ~250 chars | Medium | Overlaps with catalog line already in environment section |
| Pod Health & Failure Detection block | ~900 chars | Medium | Critical `== 1` guidance but verbose |
| PromQL Pod/Container Name Matching | ~250 chars | Low | Concise and directly useful |
| Namespace directive (`_format_namespace_directive`) | ~600 chars | Medium | Only active in namespace-scoped mode |
| Misc (exact metric names rule, catalog workflow line) | ~200 chars | Low | Minor additions |

**Total net increase:** ~4,500-5,000 characters added to the system prompt.

---

## Specific Problems

### 1. Duplicate "Trace Analysis Tools" Section (Bug)

Lines 554-563 in `base.py` contain the exact same "Trace Analysis Tools" block twice:

```python
**Trace Analysis Tools:**
- chat_tempo_tool: Conversational trace analysis - use for trace/span/latency/request flow questions
- query_tempo_tool: Direct tempo queries for specific trace searches
- get_trace_details_tool: Get detailed information about specific trace IDs

**Trace Analysis Tools:**
- chat_tempo_tool: Conversational trace analysis - use for trace/span/latency/request flow questions
- query_tempo_tool: Direct tempo queries for specific trace searches
- get_trace_details_tool: Get detailed information about specific trace IDs
```

This is a merge artifact or pre-existing duplication from `dev`. Wastes ~250 tokens and adds noise.

### 2. vLLM Block is Too Large for System Prompt (~2,200 chars)

The vLLM domain knowledge block lists every metric name, PromQL patterns, abbreviations, and examples inline. The [LLM Tool Calling Fix plan](llm-tool-calling-fix.md) specifically identifies system prompt size as the cause of Gemini's function-calling degradation. This block is the kind of content that should be discoverable via tools (e.g., `search_metrics_by_category` with category `gpu_ai`) rather than hardcoded in the prompt.

### 3. "Enhanced Metrics Catalog" Mentioned in Three Places

The catalog is referenced in:
1. The environment section: `**Enhanced Metrics Catalog**: Smart category-aware metric discovery via catalog tools`
2. The available tools section: two new tool descriptions
3. The footer: a separate "Enhanced Metrics Catalog" block with usage guidance

The plan doc flags "duplicate mentions" as a contributing factor to prompt bloat from PR #218.

---

## Risk Assessment

Based on the findings in [llm-tool-calling-fix.md](llm-tool-calling-fix.md):

- **Gemini:** High risk. The plan doc establishes that the 54% prompt increase from PR #218 pushed Gemini past the threshold where it falls back to text-based tool descriptions. The current prompt is at or above that size. The Part 1 nudge retry mitigates this at runtime but doesn't eliminate it.
- **Anthropic Haiku:** Medium risk. The `_truncate_messages()` fix prevents orphaned tool-call/result pairs (the root cause of `<function_calls>` XML leakage). Prompt size is less critical for Anthropic but still contributes to context overhead.
- **OpenAI:** Low risk. GPT-4o-mini handled the larger prompts without degradation in testing.

---

## Recommended Actions

These correspond to the Part 2 items in [llm-tool-calling-fix.md](llm-tool-calling-fix.md):

| # | Action | Expected Savings | Priority | Status |
|---|--------|-----------------|----------|--------|
| 1 | Remove duplicate "Trace Analysis Tools" block | ~250 chars | **Immediate** (bug fix) | **Done** (fixed prior to this branch) |
| 2 | Move vLLM domain knowledge out of system prompt (make it tool-discoverable) | ~2,200 chars | High | **Done** |
| 3 | Deduplicate "Enhanced Metrics Catalog" mentions (keep 1, remove 2) | ~300 chars | Medium | **Done** |
| 4 | Condense Pod Health & Failure Detection block | ~400 chars | Medium | **Done** |
| 5 | Trim tool descriptions in `prometheus_tools.py` (per plan doc Change 2) | ~200-300 tokens | Medium | Deferred |
| 6 | Summarize catalog metadata in tool results (per plan doc Change 3) | ~70-100 tokens/call | Medium | Deferred |
| 7 | Condense Intelligence Style section | ~400 chars | Medium | **Done** |
| 8 | Remove duplicate "ANSWER ONLY" line | ~70 chars | Low | **Done** |
| 9 | Remove duplicate "STRICT RULES" workflow section | ~350 chars | Medium | **Done** |
| 10 | Remove "CORE PRINCIPLES" block | ~380 chars | Medium | **Done** |

**Total reduction:** ~4,000 characters from the system prompt (~31% reduction).

---

## Completed Changes — Detailed Rationale

### Action 1: Remove duplicate "Trace Analysis Tools" block (~250 chars)
**Done** (fixed prior to this branch)

**What changed:** The identical 3-tool "Trace Analysis Tools" section appeared twice in the prompt. One copy was removed.

**Why:** This was a merge artifact from the `dev` branch. Duplicate tool listings waste tokens and add noise — the model gains nothing from seeing the same tool described twice. It can also cause confusion about whether there are actually two different sets of trace tools.

**What it achieved:** Eliminated a pure duplication bug. No behavioral impact since the remaining copy is identical.

### Action 2: Replace vLLM domain knowledge block with tool pointer (~2,100 chars)
**Done**

**What changed:** A 40-line block listing every vLLM metric name, PromQL patterns, abbreviations, and examples was replaced with:
```
**vLLM / Model-Serving Metrics:**
For vLLM metric names, PromQL patterns, and abbreviations, use `search_metrics_by_category` with category `gpu_ai`.
Key concepts: latency (TTFT, TPOT, E2E), throughput (tokens/sec, requests), KV cache utilization, prefix caching.
```

**Why:** This was the single largest contributor to prompt bloat (~2,200 chars). The [LLM Tool Calling Fix plan](llm-tool-calling-fix.md) identifies system prompt size as the root cause of Gemini's function-calling degradation — the 54% prompt increase from PR #218 pushed Gemini past the threshold where it falls back to text-based tool descriptions instead of using the function calling API.

The vLLM content is reference data (metric names, PromQL snippets) that the model needs *when answering vLLM questions*, not on every single query. The `gpu_ai` category in `src/core/metrics_catalog.py` already contains all the same metric names, descriptions, and keywords. The `search_metrics_by_category("gpu_ai")` tool call returns this data on demand.

**What it achieved:** The largest single saving (~2,100 chars). The 2-line replacement preserves the key concept vocabulary (TTFT, TPOT, E2E, KV cache, prefix caching) so the model still recognizes vLLM-related questions and knows which tool to call. The detailed metric names and PromQL patterns are now retrieved dynamically only when needed, keeping every non-vLLM conversation lighter.

### Action 3: Deduplicate "Enhanced Metrics Catalog" mentions (~300 chars)
**Done**

**What changed:** Two of three "Enhanced Metrics Catalog" references were removed:
1. The environment section line: `**Enhanced Metrics Catalog**: Smart category-aware metric discovery via catalog tools`
2. The footer block: a standalone paragraph explaining how to use catalog tools

The remaining reference — the two tool descriptions in the "Available Tools" section (`get_metrics_categories` and `search_metrics_by_category`) — was kept.

**Why:** The catalog was described in three separate places in the prompt. The tool descriptions in "Available Tools" are the most actionable reference — they tell the model exactly which tools exist and what they do. The environment line was a vague marketing-style description that added no information beyond what the tool descriptions already convey. The footer block repeated the same guidance a third time.

Triple-mentioning a feature doesn't make the model use it more reliably — it just consumes tokens that push more useful instructions further from the model's attention window.

**What it achieved:** Removed ~300 chars of redundant catalog mentions while preserving the tool descriptions that actually guide tool selection.

### Action 4: Condense Pod Health & Failure Detection block (~350 chars)
**Done**

**What changed:** The 9-line block was reduced to 3 concise bullets:
```
**Pod Health & Failure Detection:**
- `kube_pod_status_phase` only tracks pod-level phases — most failures (CrashLoopBackOff, ImagePullBackOff, OOMKilled) are NOT in "Failed" phase
- Check container-level metrics: `kube_pod_container_status_waiting_reason` and `kube_pod_container_status_terminated_reason`
- ALWAYS append `== 1` to kube-state-metrics status queries to exclude stale time series
```

**Why:** The original block contained critical operational knowledge (the multi-metric requirement and the `== 1` rule) but wrapped it in verbose explanations and inline PromQL patterns. The PromQL patterns (e.g., `kube_pod_container_status_waiting_reason{reason=~"CrashLoopBackOff|ImagePullBackOff|..."}`) are something the LLM can construct from the metric names — it doesn't need pre-built query templates in the system prompt. The explanatory text ("containers stuck in waiting state", "containers that terminated with errors") described what the metric names already convey.

**What it achieved:** Preserved the three essential rules (check container-level metrics, not just pod phase; use specific metric names; append `== 1`) while removing the verbose scaffolding. The model still knows *which* metrics to query and *why*, but the prompt is ~350 chars lighter.

### Action 5: Trim tool descriptions in `prometheus_tools.py` (Deferred)
**TODO**

**Why deferred:** Tool descriptions in `prometheus_tools.py` are passed to all LLM providers via the function calling API. Changing them affects Gemini, Anthropic, and OpenAI simultaneously. Each provider handles tool descriptions differently in terms of tokenization and attention. This needs separate evaluation and testing across all providers before modifying.

### Action 6: Summarize catalog metadata in tool results (Deferred)
**TODO**

**Why deferred:** This would change the runtime behavior of tool results (what data gets returned from `search_metrics_by_category`), not just the static system prompt. It's a behavioral change that could affect answer quality and needs its own testing cycle in a separate PR.

### Action 7: Condense Intelligence Style from 5 sub-sections to 3 (~400 chars)
**Done**

**What changed:** The "Your Intelligence Style" section was reduced from 5 detailed sub-sections (with bullet-point examples under each) to 3 concise one-line bullets:
```
1. **Rich Contextual Analysis**: Provide context, thresholds, and implications — not just raw numbers
2. **Intelligent Grouping**: Group related pods by function (AI/ML Stack, Infrastructure, Data Storage) with counts
3. **Operational Intelligence**: Include health assessments, trend context, and actionable recommendations
```

Two sub-sections were removed:
- "Always Show PromQL Queries" — redundant with the Response Format section which already specifies `**PromQL Used:** your_query_here` in the template
- "Smart Follow-up Context" — low-value guidance ("cross-reference related metrics when helpful", "provide trend context") that the model already does naturally and that conflicts with the PRIMARY RULE to not explore beyond the user's question

**Why:** The Intelligence Style section existed to shape *how* the model presents answers, but the verbose sub-bullet examples (e.g., `Group related pods: "AI/ML Stack (2 pods): llama-3-2-3b-predictor, llamastack"`) were prescriptive formatting examples, not behavioral guidance. LLMs don't need example output strings to understand "group by function" — the instruction itself is sufficient. The duplicate PromQL instruction was pure waste.

**What it achieved:** ~400 chars saved. The 3 remaining bullets preserve the core behavioral guidance (contextual analysis, grouping, operational intelligence) without the redundant examples and duplicate instructions.

### Action 8: Remove duplicate "ANSWER ONLY" line (~70 chars)
**Done**

**What changed:** Removed the standalone line:
```
**CRITICAL: ANSWER ONLY WHAT THE USER ASKS - DON'T EXPLORE EVERYTHING**
```

**Why:** This was a verbatim restatement of the PRIMARY RULE at the top of the prompt:
```
PRIMARY RULE: ANSWER ONLY WHAT THE USER ASKS. DO NOT EXPLORE BEYOND THEIR SPECIFIC QUESTION.
```

Repeating the same instruction mid-prompt doesn't reinforce it — it just consumes tokens. The PRIMARY RULE is already placed at the highest-priority position (top of prompt) with emoji emphasis.

**What it achieved:** ~70 chars saved. Minor individually, but part of a pattern of removing redundant restatements that collectively add up.

### Action 9: Remove duplicate "STRICT RULES" workflow section (~350 chars)
**Done**

**What changed:** Removed the 7-line "STRICT RULES - FOLLOW FOR ANY QUESTION" block that appeared immediately after the "Your Workflow" section.

**Why:** The "STRICT RULES" block was a point-by-point restatement of the "Your Workflow" section directly above it. Compare:

| Workflow section | STRICT RULES section |
|---|---|
| CHOOSE TOOL TYPE: Trace → chat_tempo_tool | For trace questions: Use chat_tempo_tool |
| Metrics → search_metrics + execute_promql | For metrics questions: Call search_metrics, then execute_promql |
| Alert investigations → execute_promql or korrel8r | For alert questions: Use execute_promql (ALERTS) or korrel8r tools |
| Log questions → get_correlated_logs | For log questions: Use get_correlated_logs |
| ANSWER: Provide the specific answer — DONE! | Report the specific answer — DONE! |

Every line in "STRICT RULES" mapped 1:1 to a line in "Your Workflow" with slightly different wording. Having two versions of the same routing logic risks the model weighting one over the other if they ever diverge, and wastes tokens when they're identical.

**What it achieved:** ~350 chars saved by removing a pure duplication. The "Your Workflow" section remains and provides the same tool-selection routing.

### Action 10: Remove "CORE PRINCIPLES" block (~380 chars)
**Done**

**What changed:** Removed the 5-bullet "CORE PRINCIPLES" block:
```
**CORE PRINCIPLES:**
- **BE THOROUGH BUT FOCUSED**: Use as many tools as needed to answer comprehensively
- **STOP when you have enough data** to answer the question well
- **ANSWER ONLY** what they asked for
- **NO EXPLORATION** beyond their specific question
- **BE DIRECT** - don't analyze everything about a topic
```

**Why:** All 5 bullets restate the PRIMARY RULE from the top of the prompt ("ANSWER ONLY WHAT THE USER ASKS. DO NOT EXPLORE BEYOND THEIR SPECIFIC QUESTION.") in different words. This was the third location in the prompt that said "answer only what they asked" — the PRIMARY RULE, the now-removed "ANSWER ONLY" line (Action 8), and this block. Three restatements of the same constraint provide diminishing returns and consume tokens that could be used for actual behavioral guidance.

**What it achieved:** ~380 chars saved. The PRIMARY RULE at the top of the prompt already carries the "focused, no exploration" instruction with maximum emphasis (emoji, caps, first position). Removing redundant restatements tightens the prompt without losing any behavioral signal.

---

## What Was NOT Changed

- **Tool descriptions in the "Available Tools" section**: These are the primary reference for tool selection and were kept intact
- **Tool Selection Guidelines** (trace, alert, log routing): Critical behavioral guidance, kept as-is
- **Interpreting Metrics Correctly**: Boolean/gauge/counter guidance, kept as-is
- **Always Group Metrics for Detailed Breakdowns**: Kept as-is
- **PromQL Pod/Container Name Matching**: Concise and directly useful, kept as-is
- **Response Format template**: Kept as-is (also serves as the "show PromQL" instruction, replacing the removed Intelligence Style sub-section)
- **Critical Rules**: Kept as-is
- **All function signatures and behavior**: No code changes outside the prompt string
- **Llama-specific instructions** in `llama_bot.py`: Separate from the base prompt, untouched

---

## Alternative Considered: Dynamic Per-Query Prompt via Python Classifier

An alternative approach was considered where the system prompt would be built dynamically for each query:

1. Query comes in
2. Python logic (not the LLM) classifies the query into metric category/categories
3. System prompt is built with only the sections relevant to that query
4. Each query gets a small, focused, customized prompt

### What works well about this idea

The core insight is sound: most queries only need a subset of the prompt. A vLLM latency question doesn't need Pod Health detection rules, and a "show me alerts" query doesn't need vLLM metric names. Sending only relevant sections would mean smaller prompts *and* more detail per category — you could include the full vLLM metric list for vLLM queries without penalizing every other query.

### Where it gets difficult

**1. The classifier is the hard part**

Python keyword matching is brittle. Consider these queries:

| Query | Obvious category? |
|---|---|
| "why is my model slow?" | vLLM latency? Pod health? Traces? Node resources? |
| "is everything okay?" | All categories |
| "what happened in the last hour?" | Alerts + logs + metrics |
| "show me GPU usage" | GPU metrics, but also vLLM KV cache? |
| "are there any problems with my deployment?" | Pod health + alerts + logs |

The LLM is better at this classification than keyword matching — it understands intent, not just keywords. The current approach delegates classification to the LLM via tool selection: the model reads the query, picks the right tool (`chat_tempo_tool`, `search_metrics_by_category("gpu_ai")`, `get_correlated_logs`, etc.), and the tool returns the relevant data.

The vLLM change (Action 2) is a lightweight version of this idea: instead of embedding vLLM knowledge in the static prompt, the LLM calls `search_metrics_by_category("gpu_ai")` on demand. The LLM *is* the classifier.

**2. ~~Multi-turn conversations shift categories~~ — Not applicable**

This concern was originally raised but does not apply to this architecture. Per the [chat flow](chat-flow-e2e.md) (sections 4-6), a **new chatbot instance is created for every request**. The `chat_tool.py` handler calls `create_chatbot()` fresh each time, and `_create_system_prompt()` is called on the new instance. Conversation history is passed in from the frontend as a parameter — it is not held in the chatbot's state.

This means the system prompt is already rebuilt from scratch on every single message. A per-query dynamic prompt would not cause mid-conversation instability because there is no persistent conversation-level prompt to destabilize. Each turn is independent at the chatbot layer.

This removes one of the arguments against the dynamic approach, though the remaining concerns (classifier accuracy, false negatives, maintenance) still apply.

**3. False negatives are expensive**

With a static prompt, including extra instructions costs tokens but doesn't cause failures. With a dynamic prompt, *missing* a relevant section means the model lacks guidance it needs. A false negative (query needs pod health rules but classifier didn't include them) produces a wrong answer. A false positive (including extra sections) just wastes some tokens.

The asymmetry matters: the cost of getting classification wrong is higher than the cost of a slightly larger prompt.

**4. Maintenance burden**

Instead of 1 prompt to maintain, there would be N prompt fragments plus classification logic that must stay in sync. When a new tool or metric category is added, both the fragment *and* the classifier need updating. The current approach has a single source of truth.

### Middle-ground approach (what was implemented)

A hybrid captures the benefit without the classification risk:

- **Static core** (~4,000 chars): PRIMARY RULE, tool list, workflow, response format, critical rules — always included, needed for every query
- **Dynamic detail** (on-demand via tools): Category-specific metric names, PromQL patterns, domain knowledge — retrieved by the LLM when it calls `search_metrics_by_category`

This is what Action 2 (vLLM block replacement) implements. The static prompt tells the model *what tools exist and how to use them*. The tools return *domain-specific detail* on demand. The LLM acts as its own classifier via tool selection.

If further reduction is needed, the next steps would be Actions 5 and 6 (trimming tool descriptions and summarizing catalog metadata), which reduce the static portion further while keeping the dynamic retrieval path intact — no Python classifier required.

### Comparison

| Approach | Prompt size | Classification risk | Maintenance |
|---|---|---|---|
| Static prompt (before) | ~13,100 chars | None | Low |
| Static + tool retrieval (current) | ~9,100 chars | None (LLM classifies) | Low |
| Python classifier + dynamic prompt | ~2,000-4,000 chars | Medium-high | High |

Note: Multi-turn stability was originally listed as a differentiator but is not applicable — a new chatbot instance is created per request, so the system prompt is already rebuilt from scratch each turn (see point 2 above).

### Decision

Not pursued. The current approach (static core + LLM-driven tool retrieval) already implements the useful part of this idea — moving domain knowledge into tools — without the risky part — Python-side query classification. The LLM doing its own classification via tool selection is more robust than keyword matching and doesn't require maintaining a separate classifier.

---

## Current Mitigations in Place (Part 1)

These are already present on the branch and mitigate the worst symptoms:

- **Gemini nudge retry** (`google_bot.py`): Detects text-based tool calls and sends a one-shot nudge asking the model to use function calling API. Max 1 retry per session.
- **Llama fabrication guard + text-tool-call detection** (`llama_bot.py`): Detects when Llama skips tools (fabrication) or writes tool calls as text. Nudges with `tool_choice="required"` on retry. See [Llama test results](llama-tool-calling-test-results.md) for before/after comparison.
- **Llama tool allowlist** (`llama_bot.py`): Filters 42 tools down to 19 essential ones, saving ~5,000 tokens. Uses `_get_tool_allowlist()` hook in `base.py`.
- **Llama compact system prompt** (`llama_bot.py`): Overrides `_get_base_prompt()` with a ~1,600 char version (vs ~10,600 shared), tailored for the 14K token context limit.
- **Anthropic `_truncate_messages()`** (`base.py`): Groups tool-call/result pairs atomically during truncation, preventing orphaned pairs that cause XML leakage.
- **Namespace override fix** (`base.py`): `_inject_namespace_into_promql()` replaces existing namespace filters rather than skipping injection.
