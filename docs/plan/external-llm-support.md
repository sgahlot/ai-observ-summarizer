# External LLM Support — Issues and Fixes

**Last Updated:** 2026-03-09

## Table of Contents

- [Executive Summary](#executive-summary)
- [Context](#context)
- [Completed Fixes](#completed-fixes)
  - [1. Makefile: process_llm_url appends port to fully-qualified URLs](#1-makefile-process_llm_url-appends-port-to-fully-qualified-urls)
  - [2. Makefile: Added LLM_MODEL_ID parameter](#2-makefile-added-llm_model_id-parameter)
  - [3. MCP server: Model name mapping for external vLLM](#3-mcp-server-model-name-mapping-for-external-vllm)
  - [4. llm-service chart: OOM for 70B model (memory and GPU utilization)](#4-llm-service-chart-oom-for-70b-model-memory-and-gpu-utilization)
  - [5. Makefile: Helm install timeout and atomic rollback](#5-makefile-helm-install-timeout-and-atomic-rollback)
  - [6. llm-service chart: Local chart dependency](#6-llm-service-chart-local-chart-dependency)
  - [7. llama-stack chart: Upgrade to v0.7.2 and local chart dependency](#7-llama-stack-chart-upgrade-to-v072-and-local-chart-dependency)
  - [8. LLAMA_STACK_URL: API path updated for LlamaStack 0.5.2](#8-llama_stack_url-api-path-updated-for-llamastack-052)
- [Outstanding Issues](#outstanding-issues)
  - [1. Llama-stack chart: No provider_model_id support](#1-llama-stack-chart-no-provider_model_id-support)
  - [2. MCP server: RAG_AVAILABLE cached at startup — race condition](#2-mcp-server-rag_available-cached-at-startup--race-condition)
  - [3. local-dev.sh: No support for external LLM URL or API token](#3-local-devsh-no-support-for-external-llm-url-or-api-token)
  - [4. MCP server deployment: LLM_URL env var is hardcoded](#4-mcp-server-deployment-llm_url-env-var-is-hardcoded)
- [Local 70B Model Deployment](#local-70b-model-deployment)
  - [Hardware requirements](#hardware-requirements)
  - [Issues encountered and fixes](#issues-encountered-and-fixes)
  - [Deployment timeline](#deployment-timeline)
- [Install Command Reference](#install-command-reference)
- [70B Tool-Calling and Response Quality](#70b-tool-calling-and-response-quality)
  - [Problem](#problem)
  - [Root cause analysis](#root-cause-analysis)
  - [Options investigated](#options-investigated)
    - [Option 1: Bypass LlamaStack (direct vLLM access)](#option-1-bypass-llamastack-direct-vllm-access)
    - [Option 2: Application-level mitigations](#option-2-application-level-mitigations-text-tool-call-detection--retry)
    - [Option 3: Upgrade LlamaStack to v0.7.2](#option-3-upgrade-llamastack-to-v072)
  - [Why the 70B model can't produce responses at the level of external models](#why-the-70b-model-cant-produce-responses-at-the-level-of-external-models)
- [Git Branch](#git-branch)

---

## Executive Summary

**Goal:** Enable Llama 3.3 70B (local, self-hosted) to produce response quality comparable to external models (Claude, GPT-4o) for AI observability tool-calling queries.

**Stack:** App (OpenAI SDK) -> LlamaStack (middleware) -> vLLM (model serving) -> Llama 3.3 70B FP8

### What was done

| Area | Work |
|---|---|
| **Local 70B deployment** | Deployed FP8-quantized 70B on 4x L40S GPUs (g6e.24xlarge). Fixed OOM (memory limits), CUDA OOM (GPU utilization), Helm timeout/rollback, local chart dependency. |
| **External LLM support** | Fixed URL processing for external vLLM endpoints, added `LLM_MODEL_ID` parameter, model name mapping via `serviceName`. |
| **70B chatbot class** | Created `Llama70BChatBot` with clean chat loop (no 8B guardrails), text-tool-call detection, retry logic, context cleanup. |
| **LlamaStack upgrade** | Upgraded from v0.2.22 to v0.5.2 (chart 0.5.3 -> 0.7.2). Fixed entry point, API path, model name resolution. Confirmed `tool_choice` is now forwarded to vLLM. |

### 70B response quality investigation

Three options were investigated to improve the 70B model's tool-calling and response quality:

| Option | What | Status | Result |
|---|---|---|---|
| 1. Bypass LlamaStack | Point app directly at vLLM | Blocked | KServe networking prevents direct access |
| 2. App-level mitigations | `Llama70BChatBot` with text-tool-call detection, forced retries, context cleanup | Implemented | Eliminated raw text output, reduced latency (202s -> 19s). Response quality still limited. |
| 3. Upgrade LlamaStack | v0.2.22 -> v0.5.2 to enable proper `tool_choice` forwarding | Implemented | `tool_choice` forwarding verified working via source inspection. Response quality unchanged. |

### Root cause finding

**The limitation is the model itself, not the infrastructure.**

After all three options were exhausted and the infrastructure was confirmed working correctly (`tool_choice` forwarded, vLLM tool parser active), the 70B model still:

- Completes only 1 tool call instead of chaining 2-3 (search -> query -> format)
- Ignores formatting instructions (no Technical Details, no recommendations)
- Produces bare one-liner summaries

This is a gap in Llama 3.3 70B's multi-step tool-calling capability. External models (Claude, GPT-4o) handle the same tool definitions and prompts reliably.

### Current state

- **Infrastructure:** LlamaStack upgraded to v0.5.2, `tool_choice` forwarding works, all API paths updated
- **Application:** `Llama70BChatBot` class handles 70B-specific behavior, prevents raw text tool calls from reaching users
- **Quality ceiling:** 70B responses are functional but noticeably lower quality than external models
- **Tests:** 710 tests passing
- **Branch:** All changes consolidated in `appeng-4577-support-70b-llama`

### Possible next steps

| Option | Description |
|---|---|
| Accept current quality | 70B works for basic queries; recommend external models for production use |
| Try a different model | Llama 4 or other models with stronger tool-calling support |
| Fine-tune for tool use | Train the model specifically on multi-step tool-calling sequences |
| Prompt chaining in app | Force sequential tool calls from the application side instead of relying on the model to chain them |

---

## Context

When using a Llama model that is already deployed externally (e.g., on a separate OpenShift cluster) and accessed via a URL with an API key, several issues were identified across the Makefile, Helm charts, and MCP server code. This document tracks the fixes (completed and outstanding) needed to fully support this scenario.

Additionally, deploying a 70B model locally on a GPU node required several chart and Makefile changes to handle the larger resource requirements.

**Example external use case:** A vLLM-served `llama-3-3-70b-instruct-w8a8` model accessible at:
```
https://llama-3-3-70b-instruct-w8a8-llama-3-70b-quantized.apps.ai-dev04.kni.syseng.devcluster.openshift.com/v1
```

**Example local deployment:** FP8-quantized 70B model on a g6e.24xlarge node (4x NVIDIA L40S GPUs):
```
make install LLM=llama-3-3-70b-instruct-quantization-fp8 DEV_MODE=true NAMESPACE=sgahlot-test-llama70b
```

---

## Completed Fixes

### 1. Makefile: `process_llm_url` appends port to fully-qualified URLs

**Problem:** The `process_llm_url` function checked for `:[0-9]` (a port number) to decide whether to append `:8080/v1`. Fully-qualified URLs like `https://host.example.com/` don't have a port, so they incorrectly got `:8080/v1` appended even though they were already complete.

**Fix:** Changed the grep pattern from `":[0-9]"` to `"://|:[0-9]"` so that URLs containing `://` (i.e., with a scheme) are also passed through as-is.

**File:** `Makefile`, line 131

### 2. Makefile: Added `LLM_MODEL_ID` parameter

**Problem:** When the external vLLM serves a model under a different name (e.g., `llama-3-3-70b-instruct-w8a8`) than what the llama-stack chart defaults to (e.g., `meta-llama/Llama-3.3-70B-Instruct`), there was no way to override the model ID passed to the llama-stack Helm chart.

**Fix:** Added a new `LLM_MODEL_ID` parameter to `helm_llama_stack_args` that sets `global.models.$(LLM).id` when provided.

**File:** `Makefile`, line 144

**Note:** Once the llama-stack chart supports `provider_model_id` (see Outstanding Issue 1), this parameter should be changed to set `global.models.$(LLM).providerModelId` instead of `global.models.$(LLM).id`, so that the public model name (`meta-llama/Llama-3.3-70B-Instruct`) is preserved as `model_id` while the vLLM-specific name is set as `provider_model_id`.

### 3. MCP server: Model name mapping for external vLLM

**Problem:** The MCP server's LlamaChatBot sends the model name as-is to llama-stack (e.g., `meta-llama/Llama-3.3-70B-Instruct`), but llama-stack registers the model under the vLLM name (e.g., `llama-3-3-70b-instruct-w8a8`). There is no mapping layer in the MCP server to resolve one to the other.

**Root cause:** The MODEL_CONFIG env var (generated at install time) contains the model entry with `serviceName` set to the Helm chart key (e.g., `llama-3-3-70b-instruct`), not the actual vLLM model name. The LlamaChatBot's `_extract_model_name()` returned the raw model name without consulting the model config.

**Fix (three parts):**

1. **`scripts/generate-model-config.sh`** — When `LLM_MODEL_ID` is provided (3rd argument), use it as `serviceName` instead of `MODEL_NAME`. This ensures the MODEL_CONFIG contains the correct vLLM model name mapping.

2. **`src/chatbots/llama_bot.py`** — Updated `_extract_model_name()` to look up `serviceName` from the model config (via `get_model_config()`). When `serviceName` is set, it uses that instead of the raw model name. Falls back to the original name if no mapping exists.

3. **`Makefile`** — Updated `generate-model-config` target to pass `LLM_MODEL_ID` through to the script.

**Files changed:**
- `scripts/generate-model-config.sh` (lines 36-47, 106-109)
- `src/chatbots/llama_bot.py` (lines 16, 66-83)
- `Makefile` (line 862)

### 4. llm-service chart: OOM for 70B model (memory and GPU utilization)

**Problem:** The `llama-3-3-70b-instruct-quantization-fp8` model definition in the llm-service chart's `values.yaml` was missing a `resources` block. The `inference-service.yaml` template has two code paths:
- If a model defines `resources` -> uses custom values (and injects accelerator counts)
- If no `resources` block -> falls back to hardcoded `memory: 8Gi` limits / `4Gi` requests

With only 8Gi memory, the 70B model pod was OOM-killed immediately.

Additionally, `--gpu-memory-utilization` was set to `0.95`, which left only ~2 GiB free per GPU (44.4 GiB total per L40S). During the CUDA graph capture phase (`compile_or_warm_up_model`), vLLM ran out of GPU memory:
```
CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 44.39 GiB
of which 21.31 MiB is free. 41.50 GiB is allocated by PyTorch.
```

**Fix (two changes in `deploy/helm/rag/charts/llm-service/values.yaml`):**

1. Added `resources` block to the FP8 model:
   ```yaml
   resources:
     limits:
       cpu: "2"
       memory: 150Gi
     requests:
       cpu: "1"
       memory: 80Gi
   ```

2. Reduced `--gpu-memory-utilization` from `0.95` to `0.85`, freeing ~4.4 GiB per GPU for CUDA graph capture and KV cache initialization.

**File:** `deploy/helm/rag/charts/llm-service/values.yaml` (lines 222-228, 233)

### 5. Makefile: Helm install timeout and atomic rollback

**Problem:** The `install-rag` target used `--atomic --timeout 25m`. The `--atomic` flag causes Helm to **roll back the entire release** if the install doesn't complete within the timeout. The 70B model requires 25-30 minutes just to download (~140 GB), plus additional time for weight loading, torch compilation, and CUDA graph capture — far exceeding the 25-minute timeout.

**Fix:** Removed `--atomic` and increased timeout to `60m`. Without `--atomic`, a timeout won't roll back the deployed resources. The existing `oc wait --timeout=60m inferenceservice --all` on the next line handles waiting for readiness separately.

**File:** `Makefile`, line 519

**Before:** `--atomic --timeout 25m`
**After:** `--timeout 60m`

### 6. llm-service chart: Local chart dependency

**Problem:** The llm-service chart is sourced from a remote Helm repository (`https://rh-ai-quickstart.github.io/ai-architecture-charts`). Running `helm dependency update` (called by the Makefile's `depend` target) downloads the chart from upstream and **overwrites** any local `.tgz` modifications. This made it impossible to persist chart fixes locally.

**Fix:** Changed the `repository` for `llm-service` in `deploy/helm/rag/Chart.yaml` from the remote URL to `""`. This tells Helm to use the local chart directory at `deploy/helm/rag/charts/llm-service/` instead of downloading from upstream.

**File:** `deploy/helm/rag/Chart.yaml`, line 11

**Before:** `repository: https://rh-ai-quickstart.github.io/ai-architecture-charts`
**After:** `repository: ""`

The llm-service chart was extracted from the `.tgz` into `deploy/helm/rag/charts/llm-service/` as an unpacked directory.

### 7. llama-stack chart: Upgrade to v0.7.2 and local chart dependency

**Problem:** The llama-stack chart v0.5.3 (image `llamastack/distribution-starter:0.2.22`) did not forward `tool_choice` parameters to vLLM — it converted them to prompt instructions. This prevented reliable tool calling with the 70B model.

**Fix:** Upgraded to chart v0.7.2 (image `0.5.2`) as a local chart dependency. Like llm-service, the chart was extracted to `deploy/helm/rag/charts/llama-stack/` with `repository: ""`.

**Key changes required by the upgrade:**

| Change | Details |
|---|---|
| Entry point | LlamaStack 0.5.2 changed from `python -m llama_stack.core.server.server` to `llama stack run <config>`. Added `command` override in `values.yaml`. |
| Postgres env vars | New chart auto-injects `POSTGRES_*` env vars when `pgvector.enabled: true`. Removed 5 manual env vars from `values.yaml`. |
| API path | OpenAI-compatible endpoint moved from `/v1/openai/v1/` to `/v1/`. See Completed Fix 8. |
| Model name format | Models registered as `<provider-key>/<model-id>`. Added `_resolve_model_name()` to `Llama70BChatBot`. |

**Files:** `deploy/helm/rag/Chart.yaml`, `deploy/helm/rag/charts/llama-stack/` (new directory), `deploy/helm/rag/values.yaml`

### 8. `LLAMA_STACK_URL`: API path updated for LlamaStack 0.5.2

**Problem:** LlamaStack 0.5.2 moved the OpenAI-compatible endpoint from `/v1/openai/v1/` to `/v1/`. All references to the old path needed updating.

**Fix:** Updated the default `LLAMA_STACK_URL` from `http://localhost:8321/v1/openai/v1` to `http://localhost:8321/v1` in all locations:

| File | Change |
|---|---|
| `src/core/config.py` | Default in `is_rag_available()` and module-level constant |
| `scripts/local-dev.sh` | Export statement |
| `deploy/helm/mcp-server/templates/deployment.yaml` | In-cluster default URL |
| `src/mcp_server/setup_integration.py` | Claude Desktop config templates (2 locations) |
| `src/mcp_server/integrations/claude-desktop-config.json` | Sample config |
| `tests/core/test_config_and_models.py` | Test assertion |

---

## Outstanding Issues

### 1. Llama-stack chart: No `provider_model_id` support

**Problem:** The llama-stack Helm chart configmap template only sets `model_id` for registered models. When an external vLLM serves a model under a custom name (e.g., `llama-3-3-70b-instruct-w8a8`), the `model_id` must match that name for vLLM registration to succeed. But then the MCP server / UI, which knows the model as `meta-llama/Llama-3.3-70B-Instruct`, sends that name to llama-stack and gets a 404.

The llama-stack Python package supports a `provider_model_id` field in the model spec, which allows the public `model_id` to differ from the vLLM model name. The Helm chart v0.7.2 doesn't expose this field.

**Current workaround:** For local deployments, `Llama70BChatBot._resolve_model_name()` queries `/v1/models` at startup and resolves the provider-prefixed model ID (e.g., `llama-3-3-70b-instruct-quantization-fp8/meta-llama/Llama-3.3-70B-Instruct`). For external deployments, `LLM_MODEL_ID` overrides the model ID at install time.

**Proper fix (if needed):** Update the chart's `configmap.yaml` to pass `provider_model_id` through, and change `LLM_MODEL_ID` in the Makefile to set `providerModelId` instead of `id`.

**Upstream chart repo:** https://github.com/rh-ai-quickstart/ai-architecture-charts/tree/main/llama-stack

### 2. MCP server: `RAG_AVAILABLE` cached at startup — race condition

**Problem:** The `is_rag_available()` function in `src/core/config.py` runs once at module import time and caches the result in `RAG_AVAILABLE`. It tries to reach llama-stack's `/v1/models` endpoint with a 3-second timeout. If llama-stack is not yet ready when the MCP server starts (common since both are deployed simultaneously), `RAG_AVAILABLE` is set to `False` permanently for that process lifetime.

**Error seen:**
```
Local model 'meta-llama/Llama-3.3-70B-Instruct' is not available.
RAG infrastructure is not installed or not accessible.
```

**Workaround:** Restart the MCP server pod after llama-stack is ready.

**Possible fixes:**
- **Retry with backoff:** Retry the llama-stack health check a few times during initialization.
- **Lazy evaluation:** Check `RAG_AVAILABLE` on first use with caching and a TTL so it can recover.
- **Init container dependency:** Add an init container to the MCP server that waits for llama-stack before the main container starts.

**File:** `src/core/config.py`, lines 81-94 (`is_rag_available()`), line 155 (cached assignment)

### 3. `local-dev.sh`: No support for external LLM URL or API token

**Problem:** The `scripts/local-dev.sh` script does not accept `LLM_API_TOKEN` or an external LLM URL. There is no way to point it at an external LLM endpoint for local testing — it always port-forwards to the in-cluster llama-stack.

**Note:** The `LLAMA_STACK_URL` path has been fixed (now uses `/v1` — see Completed Fix 8). This issue is specifically about supporting external LLM endpoints during local development.

**Fix required:** Add parameters for external LLM URL and API token, skip llama-stack port-forwarding when an external URL is provided.

**File:** `scripts/local-dev.sh`

### 4. MCP server deployment: `LLM_URL` env var is hardcoded

**Problem:** In the MCP server Helm deployment template, `LLM_URL` is always constructed from the model name as an internal cluster DNS name. Unlike `LLAMA_STACK_URL` (which has a conditional override), `LLM_URL` has no override path for external endpoints.

**Impact:** Low — the MCP server uses `LLAMA_STACK_URL` for inference, not `LLM_URL`.

**File:** `deploy/helm/mcp-server/templates/deployment.yaml`

---

## Local 70B Model Deployment

### Hardware requirements

The FP8-quantized 70B model (`llama-3-3-70b-instruct-quantization-fp8`) was successfully deployed on:

| Component | Spec |
|---|---|
| **Instance** | AWS g6e.24xlarge |
| **GPUs** | 4x NVIDIA L40S (48 GB each, 192 GB total) |
| **System RAM** | 768 GiB DDR4 |
| **vCPUs** | 96 (AMD EPYC 7R13) |
| **GPU Architecture** | NVIDIA Ada Lovelace |

### Issues encountered and fixes

| Issue | Root Cause | Fix |
|---|---|---|
| Pod OOM-killed (8Gi memory limit) | FP8 model had no `resources` block, fell back to hardcoded 8Gi | Added `resources` block: 150Gi limit, 80Gi request |
| CUDA OOM during graph capture | `--gpu-memory-utilization 0.95` left only 2 GiB free per GPU | Reduced to `0.85` (frees ~4.4 GiB per GPU) |
| Helm install rolled back on timeout | `--atomic --timeout 25m` — 70B download takes 30+ min | Removed `--atomic`, increased timeout to `60m` |
| `helm dependency update` overwrites local chart fixes | llm-service sourced from remote repo | Changed to `repository: ""` with local chart directory |
| Duplicate predictor pods during rolling update | Old ReplicaSet kept spawning pods | Manually scaled down old ReplicaSet |
| HuggingFace gated repo access denied | HF token didn't have access to `meta-llama/Llama-3.3-70B-Instruct` | Requested access on HuggingFace |
| LlamaStack CrashLoopBackOff after upgrade | Entry point changed to `llama stack run` in v0.5.2 | Added `command` override in values.yaml |
| 404 on LlamaStack `/v1/openai/v1/models` | API path changed to `/v1/models` in v0.5.2 | Updated `LLAMA_STACK_URL` default everywhere |
| Model not found after LlamaStack upgrade | Model ID format changed to `<provider-key>/<model-id>` | Added `_resolve_model_name()` to query `/v1/models` |

### Deployment timeline

The 70B FP8 model has a lengthy startup sequence:

| Phase | Duration | Notes |
|---|---|---|
| Model download (first time) | ~30 min | ~140 GB of safetensors from HuggingFace |
| Weight loading (from cache) | ~60s | 16.95 GiB per GPU across 4 GPUs |
| Torch compile (dynamo) | ~47s | Bytecode transform |
| Graph compilation (first time) | ~210s | Compiled and cached for subsequent restarts |
| Graph load (from cache) | ~18s | On subsequent restarts |
| CUDA graph capture | ~40s | 67 graphs captured |
| **Total (first deploy)** | **~35 min** | Dominated by download |
| **Total (subsequent restarts)** | **~3-4 min** | All caches warm |

**Note:** The KServe readiness probe (`tcp-socket :8080, delay=0s, failureThreshold=3, period=10s`) allows only ~30 seconds before marking the pod as unhealthy. The pod may restart 1-2 times before fully starting, but this is benign since all caches are preserved across restarts within the same pod lifecycle.

---

## Install Command Reference

### External LLM (with API key)

```bash
make install \
  LLM=llama-3-3-70b-instruct \
  LLM_URL=https://<external-vllm-host>/v1 \
  LLM_MODEL_ID=<vllm-model-name> \
  LLM_API_TOKEN=$LLAMA_API_TOKEN \
  DEV_MODE=true \
  NAMESPACE=<namespace>
```

**Important:**
- `LLM_URL` must include `/v1` if the vLLM models endpoint is at `/v1/models` (the llama-stack init container appends `/models` to this URL for its health check)
- `LLM_MODEL_ID` must match the model name returned by the vLLM `/v1/models` endpoint
- `LLM_API_TOKEN` is the Bearer token required by the external endpoint

### Local 70B model deployment

```bash
make install \
  LLM=llama-3-3-70b-instruct-quantization-fp8 \
  DEV_MODE=true \
  NAMESPACE=<namespace>
```

**Notes:**
- Requires a GPU node with at least 4x GPUs and 48 GB GPU memory each (e.g., g6e.24xlarge with L40S)
- First deployment takes ~35 minutes (model download)
- Subsequent deployments are much faster if the model is cached on the node
- The pod may restart 1-2 times during initial startup — this is expected behavior

---

## 70B Tool-Calling and Response Quality

**Last Updated:** 2026-03-09

### Problem

The Llama 3.3 70B model has two related issues when used via LlamaStack:

1. **Text tool calls:** The model outputs tool calls as **plain text** (e.g., `search_metrics_by_category(category_ids=["gpu_ai"], ...)`) instead of using the OpenAI function calling API. This causes the raw tool call text to be displayed as the user-facing response.

2. **Poor response quality:** Even when tool calling works, the model produces bare one-liner responses (e.g., "The KV cache usage over the last 7 days is 0%.") instead of the formatted, detailed responses that external models (Claude, GPT) produce. The model often completes only part of the required tool chain (e.g., calls `search_metrics_by_category` but never follows up with `execute_promql` to get actual data).

### Root cause analysis

The request flow is:

```
App (OpenAI SDK) -> LlamaStack (port 8321) -> vLLM (port 8080)
```

**vLLM is correctly configured** with tool-call parsing flags in `deploy/helm/rag/charts/llm-service/values.yaml`:

```yaml
llama-3-3-70b-instruct-quantization-fp8:
  args:
    - --enable-auto-tool-choice
    - --tool-call-parser
    - llama3_json
    - --chat-template
    - /chat-templates/tool_chat_template_llama3.2_json.jinja
```

These flags tell vLLM to intercept the model's raw JSON output (`{"name": "func", "parameters": {...}}`) and convert it into proper OpenAI-compatible `tool_calls` response objects. This parsing happens at the vLLM serving layer.

**LlamaStack v0.5.2 (upgraded from v0.2.22) correctly forwards `tool_choice` to vLLM.** Source code inspection of the running pod confirmed the request flow:

1. `InferenceRouter.openai_chat_completion` -- validates `tool_choice`, passes through
2. `VLLMInferenceAdapter.openai_chat_completion` -- passes to parent unchanged
3. `OpenAIMixin.openai_chat_completion` -- includes `tool_choice` in the HTTP request to vLLM

| `tool_choice` sent by app | What reaches vLLM |
|---|---|
| `"auto"` | Forwarded correctly |
| `"required"` | Forwarded correctly |
| `{"type":"function","function":{"name":"X"}}` | Forwarded correctly |

**The root cause is the model itself.** Even with `tool_choice` properly forwarded and vLLM's tool parser active, Llama 3.3 70B does not reliably chain multiple tool calls or follow complex formatting instructions.

### Options investigated

#### Option 1: Bypass LlamaStack (direct vLLM access)

**Status: Attempted — not viable**

The idea was to point the app directly at vLLM (`http://localhost:$LLAMA_MODEL_PORT_LOCALHOST/v1`) to bypass LlamaStack entirely and verify whether vLLM's `--tool-call-parser` works end-to-end.

**What was tried:**
- Modified `scripts/local-dev.sh` to set `LLAMA_STACK_URL="http://localhost:$LLAMA_MODEL_PORT_LOCALHOST/v1"`
- Ran `curl http://localhost:8080/v1/models`

**Result:** `curl: (52) Empty reply from server`. The vLLM model service is not directly reachable on plain HTTP — it's wrapped in a KServe InferenceService which uses its own networking layer (HTTPS, Host-header routing, or sidecar proxying). The port-forward connects to KServe's gateway, not to the vLLM container directly.

**Conclusion:** Cannot isolate vLLM from LlamaStack in the current deployment topology without replicating KServe's routing. Changes were reverted.

#### Option 2: Application-level mitigations (text-tool-call detection + retry)

**Status: Implemented — provides partial improvement**

A new `Llama70BChatBot` class was created in `src/chatbots/llama70b_bot.py` with a clean chat loop (no 8B guardrails) and multiple mitigations for the tool-calling issues.

**What was implemented:**

| Mitigation | Description |
|---|---|
| Text-tool-call detection | `_detect_text_tool_calls()` catches `Tool Call:` headers, `func_name(` patterns, and `{"name": "tool_name"}` JSON patterns |
| Forced `tool_choice` | Sets `tool_choice="required"` on retry to force the model to produce a real function call |
| Retry counter | Up to 3 text-tool-call cycles before fallback (covers multi-step tool chains: search -> promql -> ...) |
| Context cleanup | `messages.pop()` removes the junk text-tool-call assistant messages to keep context lean |
| Reduced tool result size | `_get_max_tool_result_length()` set to 10,000 (down from initial 16,000) to reduce context bloat |
| Targeted metric search | Model-specific instructions tell the model to use `max_results=20` to limit search result size |
| Graceful fallback | Returns user-friendly message instead of raw text tool call output when retries are exhausted |

**Iterative test results** (query: "Show KV cache usage (%) over the last 7 days"):

| Test | Time | Steps | Result | Issue |
|---|---|---|---|---|
| 1 (initial) | 7.7s | 2 | Raw text: `search_metrics_by_category(...)` | No text-tool-call detection yet |
| 2 (+ single nudge) | 16.4s | 6 | Fallback: "I wasn't able to retrieve data" | `tool_choice="required"` worked once, model reverted to text on iteration 3 |
| 3 (+ retry counter, no context cleanup) | 202s | 12 | One-liner: "The KV cache usage is 0%." | Tools worked but context bloated to ~35K; final iteration timed out at 180s |
| 4 (+ context cleanup + reduced tool results) | 19s | 6 | Partial: found metrics but didn't call `execute_promql` | Model stopped after `search_metrics_by_category` — incomplete tool chain |

**Assessment:** The mitigations reduced response time from 202s to 19s and eliminated raw text tool calls from user-facing output. However, the model still does not reliably complete the full tool chain (search -> execute_promql -> formatted response). The response quality is noticeably better than the bare one-liners but does not approach the level of external models (Claude, GPT-4o).

#### Option 3: Upgrade LlamaStack to v0.7.2

**Status: Implemented and tested — `tool_choice` forwarding confirmed working, but response quality unchanged**

Upgraded from chart v0.5.3 (image `0.2.22`) to chart v0.7.2 (image `0.5.2`). The hypothesis was that LlamaStack 0.5.x would properly forward `tool_choice` to vLLM for constrained decoding instead of converting it to prompt instructions.

**What was done:**

| Change | Details |
|---|---|
| Chart upgrade | Extracted 0.7.2 chart to `deploy/helm/rag/charts/llama-stack/` (local dependency) |
| Chart.yaml | `llama-stack` version 0.7.2, `repository: ""` |
| values.yaml | Added `command: [llama, stack, run, /app-config/config.yaml]` (new entry point), removed manual POSTGRES env vars (auto-injected by chart), added `RUN_CONFIG_PATH` |
| API path change | LlamaStack 0.5.2 moved OpenAI endpoint from `/v1/openai/v1/` to `/v1/` — updated `config.py`, `local-dev.sh`, Helm templates, integration configs |
| Model name resolution | LlamaStack 0.5.2 registers models as `<provider-key>/<model-id>` — added `_resolve_model_name()` to `Llama70BChatBot` to query `/v1/models` and cache the provider-prefixed ID |

**Issues encountered during upgrade:**

| Issue | Root Cause | Fix |
|---|---|---|
| CrashLoopBackOff (exit code 0) | LlamaStack 0.5.2 changed entry point from `python -m llama_stack.core.server.server` to `llama stack run <config>` | Added `command` override in values.yaml |
| 404 on `/v1/openai/v1/models` | OpenAI-compatible endpoint moved from `/v1/openai/v1/` to `/v1/` | Updated `LLAMA_STACK_URL` default in config.py, local-dev.sh, Helm templates |
| Model not found error | Model ID changed to `llama-3-3-70b-instruct-quantization-fp8/meta-llama/Llama-3.3-70B-Instruct` | Added `_resolve_model_name()` to query `/v1/models` at startup |

**Result: `tool_choice` forwarding works, but response quality is unchanged.**

Testing the same query ("Show KV cache usage (%) over the last 7 days") after the upgrade produced the same behavior as before:
- Model called `search_metrics_by_category` (1 tool call)
- Did not follow up with `execute_promql` to fetch actual data
- Returned a bare summary: "The KV cache usage over the last 7 days is not available... the `vllm:gpu_cache_usage_perc` metric is available"
- No Technical Details section, no PromQL query, no recommendations

**Conclusion:** The root cause is not the middleware — it is the model itself. Even with `tool_choice` properly forwarded, Llama 3.3 70B via vLLM does not reliably chain multiple tool calls or follow formatting instructions. The LlamaStack upgrade was necessary (fixes a real API limitation) but did not change the model's behavior.

### Why the 70B model can't produce responses at the level of external models

Despite being a capable 70B-parameter model, the local Llama 3.3 70B produces noticeably worse responses than external models (Claude, GPT-4o). After exhaustive investigation (Options 1-3), the root cause is the **model's own tool-calling and instruction-following capability**, not the infrastructure.

**What was ruled out:**

| Hypothesis | Investigation | Finding |
|---|---|---|
| LlamaStack converts `tool_choice` to prompt instructions | Upgraded to LlamaStack v0.5.2 (Option 3) | Confirmed: v0.5.2 forwards `tool_choice` directly to vLLM. Response quality unchanged. |
| vLLM tool-call parser not working | vLLM has `--enable-auto-tool-choice` and `--tool-call-parser llama3_json` configured | vLLM parser works — tool calls are returned as proper API objects (not text) after the LlamaStack upgrade |
| Application code issues | Created dedicated `Llama70BChatBot` with clean chat loop (Option 2) | Mitigations helped (19s vs 202s, no raw text in output) but didn't fix incomplete tool chains |

**The actual root cause: Model behavior limitations**

External models (Claude, GPT-4o) reliably:
- Chain multiple tool calls (search -> query -> format response)
- Follow formatting instructions (Technical Details, recommendations, markdown)
- Complete the full tool pipeline before generating a final response

Llama 3.3 70B via vLLM, even with `tool_choice` properly forwarded:
- Stops after one tool call (calls `search_metrics_by_category` but skips `execute_promql`)
- Ignores formatting instructions in the system prompt
- Produces bare one-liner summaries based on metric names rather than actual data

This is a **model capability gap** in multi-step tool chaining and complex instruction following. The 70B model's tool-calling support via vLLM is functional for single tool calls but unreliable for multi-step tool chains that require the model to plan a sequence of calls, process intermediate results, and produce a structured final response.

**What has been done (all three options exhausted):**

| Option | Status | Impact |
|---|---|---|
| Option 1: Bypass LlamaStack | Blocked — KServe networking prevents direct vLLM access | N/A |
| Option 2: Application-level mitigations | Implemented — `Llama70BChatBot` with text-tool-call detection, retries, context cleanup | Moderate — eliminates raw text output, reduces latency |
| Option 3: Upgrade LlamaStack to v0.7.2 | Implemented — `tool_choice` forwarding confirmed working | Necessary infrastructure fix, but did not change model behavior |

**This applies to both local and external vLLM deployments.** The `external-llm-url-support` branch allows pointing to a Llama 70B model hosted on a remote cluster. However, the model behavior limitation is the same regardless of where vLLM runs — it is the model's own tool-calling capability, not the serving infrastructure.

**Bottom line:** All infrastructure-level options have been exhausted. The 70B model produces functional but lower-quality responses compared to native tool-calling providers (Claude, GPT-4o). The gap is in the model itself — multi-step tool chaining and complex formatting instructions are beyond what Llama 3.3 70B reliably handles via vLLM. Closing this gap would require either a more capable open-source model or fine-tuning the existing model for multi-step tool use.

---

## Git Branch

All changes are consolidated in a single branch: **`appeng-4577-support-70b-llama`**

This branch includes:

| Category | Changes |
|---|---|
| **Helm charts** | llm-service chart with 70B memory/GPU fixes (local dependency), llama-stack v0.7.2 chart (local dependency) |
| **Application code** | `Llama70BChatBot` class, factory routing for 70B, `_resolve_model_name()` for LlamaStack 0.5.2 |
| **Configuration** | `LLAMA_STACK_URL` path updated to `/v1`, LlamaStack values.yaml (command override, env vars) |
| **Tests** | 710 tests passing, mocks for `_resolve_model_name` |
| **Makefile** | Removed `--atomic`, increased timeout to 60m |

Previously, changes were split across `local-70b-model-support` and `external-llm-url-support`. Those branches are now superseded by this consolidated branch.
