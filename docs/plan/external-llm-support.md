# External LLM Support — Issues and Fixes

**Last Updated:** 2026-03-06

## Table of Contents

- [Context](#context)
- [Completed Fixes](#completed-fixes)
  - [1. Makefile: process_llm_url appends port to fully-qualified URLs](#1-makefile-process_llm_url-appends-port-to-fully-qualified-urls)
  - [2. Makefile: Added LLM_MODEL_ID parameter](#2-makefile-added-llm_model_id-parameter)
  - [3. MCP server: Model name mapping for external vLLM](#3-mcp-server-model-name-mapping-for-external-vllm)
  - [4. llm-service chart: OOM for 70B model (memory and GPU utilization)](#4-llm-service-chart-oom-for-70b-model-memory-and-gpu-utilization)
  - [5. Makefile: Helm install timeout and atomic rollback](#5-makefile-helm-install-timeout-and-atomic-rollback)
  - [6. llm-service chart: Local chart dependency](#6-llm-service-chart-local-chart-dependency)
- [Outstanding Issues](#outstanding-issues)
  - [1. Llama-stack chart: No provider_model_id support](#1-llama-stack-chart-no-provider_model_id-support)
  - [2. Makefile: Update LLM_MODEL_ID to set providerModelId instead of id](#2-makefile-update-llm_model_id-to-set-providermodelid-instead-of-id)
  - [3. MCP server: RAG_AVAILABLE cached at startup — race condition](#3-mcp-server-rag_available-cached-at-startup--race-condition)
  - [4. local-dev.sh: No support for external LLM URL or API token](#4-local-devsh-no-support-for-external-llm-url-or-api-token)
  - [5. MCP server deployment: LLM_URL env var is hardcoded](#5-mcp-server-deployment-llm_url-env-var-is-hardcoded)
- [Local 70B Model Deployment](#local-70b-model-deployment)
  - [Hardware requirements](#hardware-requirements)
  - [Issues encountered and fixes](#issues-encountered-and-fixes)
  - [Deployment timeline](#deployment-timeline)
- [Install Command Reference](#install-command-reference)

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

---

## Outstanding Issues

### 1. Llama-stack chart: No `provider_model_id` support

**Problem:** The llama-stack Helm chart configmap template only sets `model_id` for registered models. When an external vLLM serves a model under a custom name (e.g., `llama-3-3-70b-instruct-w8a8`), the `model_id` must match that name for vLLM registration to succeed. But then the MCP server / UI, which knows the model as `meta-llama/Llama-3.3-70B-Instruct`, sends that name to llama-stack and gets a 404.

The llama-stack Python package supports a `provider_model_id` field in the model spec, which allows the public `model_id` to differ from the vLLM model name. But the Helm chart doesn't expose this field, and llama-stack 0.5.x silently ignores it.

**Fix required (two parts):**

1. **Upgrade llama-stack** to a version that supports `provider_model_id` in the vLLM provider.

2. **Update the chart's `configmap.yaml`** to pass `provider_model_id` through:
   ```yaml
         - metadata: {}
           model_id: {{ $model.id }}
   {{- if $model.providerModelId }}
           provider_model_id: {{ $model.providerModelId }}
   {{- end }}
           provider_id: {{ $key }}
           model_type: llm
   ```

**Upstream chart repo:** https://github.com/rh-ai-quickstart/ai-architecture-charts/tree/main/llama-stack

### 2. Makefile: Update `LLM_MODEL_ID` to set `providerModelId` instead of `id`

**Problem:** Currently `LLM_MODEL_ID` overrides `global.models.$(LLM).id`, which replaces the public model name entirely. Once the llama-stack chart supports `provider_model_id`, this should set `providerModelId` instead.

**Fix required:** In `Makefile`, change:
```makefile
$(if $(LLM_MODEL_ID),--set global.models.$(LLM).id='$(LLM_MODEL_ID)',) \
```
to:
```makefile
$(if $(LLM_MODEL_ID),--set global.models.$(LLM).providerModelId='$(LLM_MODEL_ID)',) \
```

**Depends on:** Outstanding Issue 1.

### 3. MCP server: `RAG_AVAILABLE` cached at startup — race condition

**Problem:** The `is_rag_available()` function in `src/core/config.py` runs once at module import time and caches the result in `RAG_AVAILABLE`. It tries to reach llama-stack's `/models` endpoint with a 3-second timeout. If llama-stack is not yet ready when the MCP server starts (common since both are deployed simultaneously), `RAG_AVAILABLE` is set to `False` permanently for that process lifetime.

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

### 4. `local-dev.sh`: No support for external LLM URL or API token

**Problem:** The `scripts/local-dev.sh` script hardcodes `LLAMA_STACK_URL` to `http://localhost:8321/v1/openai/v1` and does not accept `LLM_API_TOKEN`. There is no way to point it at an external LLM endpoint for local testing.

**Fix required:** Add parameters for external LLM URL and API token, skip llama-stack port-forwarding when an external URL is provided.

**File:** `scripts/local-dev.sh`

### 5. MCP server deployment: `LLM_URL` env var is hardcoded

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
