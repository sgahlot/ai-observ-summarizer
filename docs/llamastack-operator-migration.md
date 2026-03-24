# LlamaStack Operator Migration Plan

**Last Updated:** 2026-03-23

## Problem Statement

### For the Uninitiated

The project currently deploys LlamaStack (the AI model server) using a traditional Helm chart -- essentially a recipe that tells Kubernetes "create these exact resources." This works, but it means the Helm chart is responsible for the entire lifecycle: creating the deployment, managing updates, handling failures, etc.

The migration moves to an **operator-based** model, where a dedicated Kubernetes controller (the LlamaStack Operator) manages LlamaStack instances. Instead of telling Kubernetes exactly what to create, you tell the operator "I want a LlamaStack instance with these settings" via a Custom Resource (CR), and the operator figures out the details. Think of it as the difference between giving someone step-by-step cooking instructions vs. telling a chef "I'd like a medium-rare steak with a side salad" -- the chef (operator) knows how to handle the details, recovery, and health checks.

This aligns with the RHOAI platform direction, where `DataScienceCluster` already supports `llamastackoperator` as a managed component.

### Technical Description

The project deploys LlamaStack via the `llama-stack` Helm chart (v0.5.3) from `ai-architecture-charts`, which creates raw Kubernetes resources (Deployment, Service, ConfigMap, PVC). The migration replaces this with a `LlamaStackDistribution` Custom Resource (`llamastack.io/v1alpha1`) that is reconciled by the LlamaStack Kubernetes Operator into the same underlying resources. The operator provides lifecycle management (status reporting via CR conditions, automated reconciliation), and aligns with the RHOAI `DataScienceCluster` CR where `llamastackoperator` is a managed component. The `LLAMA_STACK_URL` HTTP endpoint consumed by all Python backend code remains unchanged.

---

## Table of Contents

- [Problem Statement](#problem-statement)
  - [For the Uninitiated](#for-the-uninitiated)
  - [Technical Description](#technical-description)
- [Context](#context)
- [Cluster Preparation (RHOAI + Prerequisite Operators)](#cluster-preparation-rhoai--prerequisite-operators)
  - [Cluster Baseline](#cluster-baseline)
  - [What NOT to Install (Already Handled by make install)](#what-not-to-install-already-handled-by-make-install)
  - [Step 1: Install GPU Operators (NFD + NVIDIA)](#step-1-install-gpu-operators-nfd--nvidia)
  - [Step 2: Install RHOAI Operator](#step-2-install-rhoai-operator)
  - [Step 3: Create DataScienceCluster with LlamaStack Operator Enabled](#step-3-create-datasciencecluster-with-llamastack-operator-enabled)
  - [Step 4: Verify LlamaStack Operator Is Running](#step-4-verify-llamastack-operator-is-running)
  - [Operators NOT Required](#operators-not-required)
  - [Architecture Charts Still Needed](#architecture-charts-still-needed)
  - [Summary: What Gets Installed Where](#summary-what-gets-installed-where)
  - [Issues Encountered](#issues-encountered)
- [Current Architecture](#current-architecture)
- [Target Architecture](#target-architecture)
- [What Does NOT Change](#what-does-not-change)
- [Migration Steps](#migration-steps)
  - [Phase 1: Add the LlamaStack Operator Chart](#phase-1-add-the-llamastack-operator-chart)
  - [Phase 2: Create LlamaStack Instance Chart (Replace llama-stack Dependency)](#phase-2-create-llamastack-instance-chart-replace-llama-stack-dependency)
  - [Phase 3: Update RAG Chart to Use the New Instance Chart](#phase-3-update-rag-chart-to-use-the-new-instance-chart)
  - [Phase 4: Update Makefile](#phase-4-update-makefile)
  - [Phase 5: Update Service References](#phase-5-update-service-references)
  - [Phase 6: Update Local Development](#phase-6-update-local-development)
  - [Phase 7: Testing and Validation](#phase-7-testing-and-validation)
- [Key Decisions](#key-decisions)
- [Risk Assessment](#risk-assessment)
- [Reference: lls-observability Operator Model](#reference-lls-observability-operator-model)

---

## Context

The project currently deploys LlamaStack as a standard Kubernetes Deployment via the `llama-stack` Helm chart (v0.5.3) from the `rh-ai-quickstart/ai-architecture-charts` repository. The goal is to migrate to the **LlamaStack Operator** model, where a Kubernetes operator manages LlamaStack instances via `LlamaStackDistribution` Custom Resources (CRD: `llamastack.io/v1alpha1`).

The `rh-ai-quickstart/lls-observability` repository is the reference implementation for this operator-based approach.

### Decision: RHOAI-managed LlamaStack Operator (Option C)

Based on cluster investigation (2026-03-23), we are proceeding with **Option C from [Key Decisions](#key-decisions)**: use the RHOAI-managed LlamaStack Operator. This means:

- RHOAI installs and manages the LlamaStack operator as a component of the `DataScienceCluster`
- No need to install a standalone operator chart (Phase 1 of the migration steps below becomes "install RHOAI + enable the component" instead of "install a Helm chart for the operator")
- The `LlamaStackDistribution` CRD is provided by RHOAI, not by a separate Helm chart
- The llm-service (vLLM model serving) continues to use the architecture charts -- only LlamaStack itself moves to the operator

### Why migrate?

- Operator-managed lifecycle (the operator handles Deployment creation, updates, and health)
- Alignment with the RHOAI platform direction (RHOAI `DataScienceCluster` already has `llamastackoperator` as a managed component)
- Standardized CR-based configuration instead of raw Helm chart values
- Status reporting via CR conditions (phase: Pending/Initializing/Ready/Failed)

---

## Cluster Preparation (RHOAI + Prerequisite Operators)

> **Status: COMPLETE** (performed 2026-03-23 on ROSA cluster `sandip-test1`)

This section describes the one-time cluster setup required before the code migration (Phases 1-7 below) can proceed. All steps here are manual operator installations and configurations performed by a cluster admin.

### Cluster Baseline

As verified on 2026-03-23 (pre-installation baseline -- all items below were subsequently installed during cluster prep):

| Property | Value |
|----------|-------|
| OCP Version | 4.21.5 |
| Cluster Type | Managed ROSA (Red Hat OpenShift on AWS) |
| RHOAI Installed | No |
| LlamaStack Operator Installed | No |
| Service Mesh Installed | No |
| Serverless / KNative Installed | No |
| GPU Operators Installed | No |
| Available Catalogs | `redhat-operators`, `certified-operators`, `community-operators`, `redhat-marketplace` |
| RHOAI Package Available | Yes (`rhods-operator`, channels: `stable-3.3`, `stable-3.x`, `fast-3.x`, etc.) |

### What NOT to Install (Already Handled by `make install`)

Our `make install` target (via `scripts/operator-manager.sh`) already installs the following **5 operators**. Do NOT install these during cluster prep -- they will be installed when the project is deployed:

| Operator | Namespace | Installed By |
|----------|-----------|-------------|
| Cluster Observability Operator | `openshift-cluster-observability-operator` | `make install-operators` |
| OpenTelemetry Operator (Red Hat build) | `openshift-opentelemetry-operator` | `make install-operators` |
| Tempo Operator (pinned to v0.19.0-3) | `openshift-tempo-operator` | `make install-operators` |
| OpenShift Logging Operator | `openshift-logging` | `make install-operators` |
| Loki Operator | `openshift-operators-redhat` | `make install-operators` |

These are observability-focused operators and have no overlap with RHOAI prerequisites.

### Step 1: Install GPU Operators (NFD + NVIDIA)

GPU operators are a prerequisite for both RHOAI and for running vLLM model serving. Both are available in the cluster's catalog.

**1a. Node Feature Discovery (NFD) Operator**

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: openshift-nfd
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: openshift-nfd
  namespace: openshift-nfd
spec:
  targetNamespaces:
  - openshift-nfd
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: nfd
  namespace: openshift-nfd
spec:
  channel: stable
  installPlanApproval: Automatic
  name: nfd
  source: redhat-operators
  sourceNamespace: openshift-marketplace
EOF
```

Wait for the operator to be ready, then create the NFD instance:

```bash
oc wait --for=condition=CatalogSourcesUnhealthy=False subscription/nfd -n openshift-nfd --timeout=120s

cat <<'EOF' | oc apply -f -
apiVersion: nfd.openshift.io/v1
kind: NodeFeatureDiscovery
metadata:
  name: nfd-instance
  namespace: openshift-nfd
spec:
  operand:
    image: registry.redhat.io/openshift4/ose-node-feature-discovery-rhel9:v4.21
    servicePort: 12000
  workerConfig:
    configData: |
      core:
        sleepInterval: 60s
EOF
```

**1b. NVIDIA GPU Operator**

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: nvidia-gpu-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: nvidia-gpu-operator
  namespace: nvidia-gpu-operator
spec:
  targetNamespaces:
  - nvidia-gpu-operator
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: gpu-operator-certified
  namespace: nvidia-gpu-operator
spec:
  channel: v24.9
  installPlanApproval: Automatic
  name: gpu-operator-certified
  source: certified-operators
  sourceNamespace: openshift-marketplace
EOF
```

Wait for the operator, then create a ClusterPolicy using the operator's built-in defaults:

```bash
oc wait csv -n nvidia-gpu-operator \
  -l operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator \
  --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s

# Extract the default ClusterPolicy from the CSV's alm-examples and apply it
GPU_CSV=$(oc get csv -n nvidia-gpu-operator -o name | grep gpu-operator-certified)
oc get $GPU_CSV -n nvidia-gpu-operator -o jsonpath='{.metadata.annotations.alm-examples}' \
  | python3 -c "import sys,json; [print(json.dumps(e)) for e in json.load(sys.stdin) if e['kind']=='ClusterPolicy']" \
  | oc apply -f -
```

**Important:** Do not create a simplified `ClusterPolicy` YAML manually. The NVIDIA GPU Operator requires many fields (`daemonsets`, `dcgm`, `gfd`, `nodeStatusExporter`, etc.). The `alm-examples` extraction above provides the complete, validated default. A hand-crafted minimal ClusterPolicy will be rejected by the operator's webhook for missing required fields.

**Verification:**

```bash
# Check NFD has detected GPU nodes
oc get nodes -l feature.node.kubernetes.io/pci-10de.present=true

# Check NVIDIA GPU operator pods are running
oc get pods -n nvidia-gpu-operator

# Check GPUs are allocatable
oc describe node <gpu-node-name> | grep nvidia.com/gpu
```

### Step 2: Install RHOAI Operator

RHOAI 3.3 is the latest GA version and supports OCP 4.21. Install via the `stable-3.3` channel.

**Note:** RHOAI 3.4 is Early Access (not GA). In RHOAI 3.3, the LlamaStack operator is fully functional but managed exclusively via CLI/YAML -- there is no "Gen AI Studio" UI in the dashboard for creating LlamaStackDistribution instances. The Gen AI Studio dashboard UI is an EA feature in RHOAI 3.4+.

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
spec: {}
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
spec:
  channel: stable-3.3
  installPlanApproval: Automatic
  name: rhods-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
EOF
```

Wait for the operator to install:

```bash
oc wait --for=condition=CatalogSourcesUnhealthy=False subscription/rhods-operator -n redhat-ods-operator --timeout=300s

# Verify CSV is in Succeeded phase
oc get csv -n redhat-ods-operator | grep rhods
```

### Step 3: Create DataScienceCluster with LlamaStack Operator Enabled

Once RHOAI is installed, create a `DataScienceCluster` with the LlamaStack operator component set to `Managed`. KServe is configured for RawDeployment mode (no Serverless/ServiceMesh dependency).

```bash
cat <<'EOF' | oc apply -f -
apiVersion: datasciencecluster.opendatahub.io/v1
kind: DataScienceCluster
metadata:
  name: default-dsc
spec:
  components:
    codeflare:
      managementState: Removed
    dashboard:
      managementState: Managed
    datasciencepipelines:
      managementState: Removed
    kserve:
      defaultDeploymentMode: RawDeployment
      managementState: Managed
      serving:
        managementState: Removed
        name: knative-serving
    llamastackoperator:
      managementState: Managed
    modelmeshserving:
      managementState: Removed
    ray:
      managementState: Removed
    trustyai:
      managementState: Removed
    workbenches:
      managementState: Removed
EOF
```

Key points about this configuration:
- **`llamastackoperator: Managed`** -- enables the LlamaStack operator; RHOAI deploys it in `redhat-ods-applications`
- **`kserve.defaultDeploymentMode: RawDeployment`** -- uses raw Kubernetes Deployments, no Knative/Serverless needed
- **`kserve.serving.managementState: Removed`** -- explicitly disables Knative Serving
- **`dashboard: Managed`** -- enables the RHOAI dashboard, but in RHOAI 3.3 it does **not** include "Gen AI Studio" for managing LlamaStack instances. LlamaStackDistribution CRs must be created via CLI (`oc apply`). The Gen AI Studio UI is available only in RHOAI 3.4+ (EA).
- Other components (`codeflare`, `datasciencepipelines`, `ray`, `trustyai`, `workbenches`) are set to `Removed` since they are not needed for this project

### Step 4: Verify LlamaStack Operator Is Running

```bash
# Check the LlamaStack operator pod is running in the RHOAI namespace
oc get pods -n redhat-ods-applications -l app.kubernetes.io/name=llama-stack-k8s-operator

# Check the LlamaStackDistribution CRD is registered
oc get crd llamastackdistributions.llamastack.io

# Check the DSC component readiness
oc get datasciencecluster default-dsc -o jsonpath='{.status.conditions[?(@.type=="LlamaStackOperatorReady")].status}'
# Expected: True

# Check no ServiceMesh/Serverless was needed
oc get ns istio-system 2>/dev/null || echo "No istio-system (expected with RawDeployment)"
oc get ns knative-serving 2>/dev/null || echo "No knative-serving (expected with RawDeployment)"
```

Expected: A pod named `llama-stack-k8s-operator-controller-manager-*` in `Running` state, the CRD `llamastackdistributions.llamastack.io` present, and `LlamaStackOperatorReady: True` in the DSC conditions.

### Operators NOT Required

Based on research, the following operators are **not needed** for our use case:

| Operator | Why Not Needed |
|----------|---------------|
| **OpenShift Serverless (KNative)** | KServe Serverless mode was deprecated in RHOAI 2.25 and retired in 3.0. We use RawDeployment mode. |
| **OpenShift Service Mesh 2.x** | Was required for KServe Serverless mode. Not needed with RawDeployment. |
| **OpenShift Service Mesh 3.x** | Optional enhancement for AI inference traffic (Gateway API Inference Extension). Not a prerequisite for LlamaStack or RHOAI core. |
| **cert-manager Operator** | Required by Service Mesh 3.x and Kueue. Since we don't use either, not needed. If RHOAI's own installation requires it, the RHOAI operator will report the dependency. |
| **Kiali Operator** | Service Mesh 2.x visualization dependency. Not applicable. |
| **Jaeger Operator** | Service Mesh 2.x tracing dependency. Not applicable (we use Tempo via our own operator install). |
| **Authorino Operator** | Token authorization for KServe endpoints. May be needed if exposing model endpoints externally with auth. Install later if required. |
| **PostgreSQL Operator** | RHOAI 3.x docs mention PostgreSQL is required for LlamaStack. However, our project deploys pgvector via the architecture charts Helm chart (`deploy/helm/rag/charts/pgvector`), which is a standalone PostgreSQL+pgvector deployment -- not operator-managed. This should satisfy the LlamaStack requirement. Verify after deployment; install a PostgreSQL operator only if the LlamaStack operator explicitly requires an operator-managed PostgreSQL instance. |

### Architecture Charts Still Needed

Even after migrating LlamaStack to the RHOAI operator, the following architecture charts from `rh-ai-quickstart/ai-architecture-charts` are **still needed**:

| Chart | Version | Purpose | Why Still Needed |
|-------|---------|---------|-----------------|
| `llm-service` | v0.5.4 | vLLM model serving (InferenceService/ServingRuntime via KServe) | LlamaStack operator does not deploy models -- it connects to a pre-deployed vLLM endpoint. The llm-service chart deploys the model. |
| `pgvector` | v0.5.0 | PostgreSQL with pgvector extension | Vector database for RAG. LlamaStack operator supports pgvector as a backend but does not deploy it. |

The only chart being **removed** is:

| Chart | Version | Replaced By |
|-------|---------|------------|
| `llama-stack` | v0.5.3 | RHOAI LlamaStack Operator (`LlamaStackDistribution` CR) |

### Summary: What Gets Installed Where

```
Cluster Prep (manual, one-time)         Our Project (make install)
========================================  ========================================
1. NFD Operator                          1. Cluster Observability Operator
2. NVIDIA GPU Operator                   2. OpenTelemetry Operator
3. RHOAI Operator (stable-3.3)          3. Tempo Operator
   +-- DataScienceCluster               4. OpenShift Logging Operator
       +-- llamastackoperator: Managed   5. Loki Operator
       +-- kserve: RawDeployment
   +-- Manual Route (ROSA/OSD only,      6. Observability stack (Helm)
       see Issues Encountered)
                                         7. MCP server (Helm)
                                         8. RAG (Helm):
                                            - llm-service (arch chart)
                                            - LlamaStackDistribution CR (NEW)
                                            - pgvector (arch chart)
                                         9. React UI / Console Plugin (Helm)
```

### Issues Encountered

#### Issue 1: RHOAI Dashboard 503 on Managed Clusters (ROSA/OSD)

**Symptom:** After installing RHOAI 3.3 and creating the DataScienceCluster, the RHOAI dashboard at `https://data-science-gateway.apps.<cluster>/` returns HTTP 503 (Service Unavailable).

**Root cause:** RHOAI 3.x uses the Gateway API with an Istio-based gateway in the `openshift-ingress` namespace. As part of setup, the RHOAI operator attempts to create a `NetworkPolicy` named `kube-auth-proxy` in the `openshift-ingress` namespace. On managed clusters (ROSA/OSD), the admission webhook `networkpolicies-validation.managed.openshift.io` blocks this because `openshift-ingress` is a Red Hat-managed namespace.

The `GatewayConfig` status shows:
```
admission webhook "networkpolicies-validation.managed.openshift.io" denied the request:
User 'system:serviceaccount:redhat-ods-operator:redhat-ods-operator-controller-manager'
prevented from creating network policy that may impact default ingress, which is managed by Red Hat.
```

The gateway pod and auth-proxy pod deploy successfully, but no OpenShift Route is created to expose them externally.

**Fix:** Manually create an OpenShift Route in the `openshift-ingress` namespace pointing to the **Istio gateway service** (not the auth proxy -- the auth proxy is invoked by the gateway via an Istio `EnvoyFilter` for `ext_authz`):

```bash
cat <<'EOF' | oc apply -f -
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: rhods-dashboard
  namespace: openshift-ingress
  labels:
    app: rhods-dashboard
spec:
  host: data-science-gateway.apps.<cluster-domain>
  to:
    kind: Service
    name: data-science-gateway-data-science-gateway-class
    weight: 100
  port:
    targetPort: 443
  tls:
    termination: reencrypt
    insecureEdgeTerminationPolicy: Redirect
  wildcardPolicy: None
EOF
```

Replace `<cluster-domain>` with your cluster's base domain (e.g., `sandip-test1.k0ih.p1.openshiftapps.com`).

**Important:** The traffic flow is: Route -> Istio Gateway (port 443) -> EnvoyFilter (`data-science-authn-filter`) calls auth proxy for authentication -> Gateway forwards to dashboard via HTTPRoute. If you mistakenly point the Route at the `kube-auth-proxy` service directly, you'll see a blank page showing only "Authenticated" because the auth proxy's upstream is `static://200` (auth-only mode, not a reverse proxy).

**Post-login note:** After creating the Route, clear browser cookies or use an incognito window to avoid stale OAuth session cookies.

**Verification:** `curl -sk -o /dev/null -w '%{http_code}' https://data-science-gateway.apps.<cluster-domain>/` should return `302` (redirect to OAuth login).

#### Issue 2: MCP Server Helm Upgrade Fails with CA Bundle ConfigMap Conflict

**Symptom:** `make install-mcp-server` fails on upgrade with:

```
conflict occurred while applying object ai-observability/aiobs-mcp-server-trusted-ca-bundle /v1, Kind=ConfigMap:
Apply failed with 1 conflict: conflict with "service-ca-operator" using v1: .data.service-ca.crt
```

**Root cause:** The MCP server Helm chart (`deploy/helm/mcp-server/templates/configmap-ca.yaml`) creates a ConfigMap with `service.beta.openshift.io/inject-cabundle: "true"` annotation and also declares `data: service-ca.crt: ""`. The OpenShift `service-ca-operator` detects the annotation and populates `service-ca.crt` with the cluster CA certificate, taking ownership of that field.

This only fails when **Helm 4 server-side apply (SSA)** is in use. Helm 4 defaults to SSA for **new** releases, but latches to client-side apply for releases originally created under Helm 3 (via `--server-side "auto"`). This is why the same chart works on one cluster but fails on another:

| Cluster | Release first created | `APPLY_METHOD` (from `helm get metadata`) | Result on upgrade |
|---|---|---|---|
| Old (tsisodia-dev, OCP 4.20) | Feb 2026 (Helm 3 era) | `client-side apply (defaulted)` | Works -- no field ownership enforcement |
| New (sandip-test1, OCP 4.21) | Mar 2026 (Helm 4) | `server-side apply` | Fails -- SSA enforces field ownership, Helm and service-ca-operator conflict on `data.service-ca.crt` |

The K8s `managedFields` confirm this -- on the new cluster, Helm's operation is `Apply` while on the old cluster it is `Update`.

**Fix:** Remove the `data` block from the ConfigMap template. The annotation is sufficient -- the `service-ca-operator` will populate `service-ca.crt` automatically. Helm should not declare a field that another controller manages:

```yaml
# deploy/helm/mcp-server/templates/configmap-ca.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Values.trustedCA.configMapName | default "aiobs-mcp-server-trusted-ca-bundle" }}
  annotations:
    service.beta.openshift.io/inject-cabundle: "true"
# No data section -- service-ca-operator injects service-ca.crt via the annotation
```

**Workaround (without chart change):** Delete the ConfigMap before upgrading so Helm recreates it fresh:

```bash
oc delete configmap aiobs-mcp-server-trusted-ca-bundle -n <namespace> --ignore-not-found
make install-mcp-server ...
```

**Note:** This pattern applies to any Helm chart that creates ConfigMaps with `inject-cabundle: "true"` and also declares the injected key in `data`. With the industry migration to Helm 4 (Helm 3 EOL: November 2026), all such charts will need this fix for new installations.

#### Issue 3: LlamaStack Operator NetworkPolicy Blocks Same-Namespace Consumers

**Symptom:** After deploying the RAG chart, the MCP server (and alerting cronjob) cannot reach LlamaStack. LLM API requests time out after 180 seconds. Direct `curl` from the MCP server pod to the LlamaStack pod IP also times out.

**Root cause:** The LlamaStack operator creates a `NetworkPolicy` (`llamastack-network-policy`) that restricts ingress to the LlamaStack pod on port 8321 to only:

1. Pods with label `app.kubernetes.io/part-of: llama-stack` (from any namespace)
2. Pods from the `redhat-ods-applications` namespace (where the operator runs)

```yaml
# Operator-created NetworkPolicy (abbreviated)
spec:
  podSelector:
    matchLabels:
      app: llama-stack
      app.kubernetes.io/instance: llamastack
  ingress:
  - from:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          app.kubernetes.io/part-of: llama-stack
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: redhat-ods-applications
    ports:
    - port: 8321
```

The MCP server and alerting cronjob are legitimate consumers but don't carry the `app.kubernetes.io/part-of: llama-stack` label, so they are blocked.

**Fix:** Add an additional `NetworkPolicy` in the `llama-stack-instance` subchart (`deploy/helm/rag/llama-stack-instance/templates/networkpolicy.yaml`) that allows all pods in the same namespace to reach LlamaStack on port 8321. Kubernetes NetworkPolicies are additive -- if any policy allows the traffic, it is permitted. This approach:

- Does not modify or conflict with the operator-managed NetworkPolicy
- Does not require adding misleading labels (like `part-of: llama-stack`) to consumer pods
- Won't be undone by the operator reconciling its resources (since it's a separate resource)
- Is versioned and managed by our Helm chart

```yaml
# Our additive NetworkPolicy (llamastack-allow-namespace)
spec:
  podSelector:
    matchLabels:
      app: llama-stack
      app.kubernetes.io/instance: llamastack
  ingress:
  - from:
    - podSelector: {}    # All pods in the same namespace
    ports:
    - port: 8321
```

---

## Current Architecture

```
Helm install (make install-rag)
  |
  +-- rag chart (deploy/helm/rag/)
       |
       +-- llama-stack v0.5.3           <-- from ai-architecture-charts
       |     Creates: Deployment, Service, ConfigMap, PVC, Secret
       |     Image: llamastack/distribution-starter:0.6.0
       |     Service: llamastack:8321
       |
       +-- llm-service v0.5.4           <-- local subchart
       |     Creates: ServingRuntime, InferenceService (KServe)
       |     Runs: vLLM model servers
       |
       +-- pgvector v0.5.0              <-- from ai-architecture-charts
             Creates: Deployment, Service, Secret
             Runs: PostgreSQL with pgvector extension
```

**Key files involved:**

| File | Role |
|------|------|
| `deploy/helm/rag/Chart.yaml` | Declares `llama-stack` as dependency from ai-architecture-charts |
| `deploy/helm/rag/Chart.lock` | Pins `llama-stack` at v0.5.3 |
| `deploy/helm/rag/values.yaml` | Configures `llama-stack` env vars, MCP servers, model settings |
| `deploy/helm/rag/charts/llm-service/` | Local bundled llm-service chart |
| `deploy/helm/mcp-server/templates/deployment.yaml` | Hardcodes `LLAMA_STACK_URL` to `llamastack.{ns}.svc.cluster.local:8321` |
| `deploy/helm/alerting/templates/cronjob.yaml` | Hardcodes `LLAMA_STACK_URL` to `llamastack.{ns}.svc.cluster.local:8321` |
| `Makefile` | `helm_llama_stack_args` function, `install-rag` target, `depend` target |
| `scripts/local-dev.sh` | Port-forwards to LlamaStack service by label |

---

## Target Architecture

### Option C: RHOAI-managed operator (recommended)

```
Cluster Prep: RHOAI DataScienceCluster (one-time, see Cluster Preparation above)
  |
  +-- LlamaStack Operator (managed by RHOAI)
       Namespace: redhat-ods-applications
       Image: managed by RHOAI (not user-configurable)
       CRD: llamastackdistributions.llamastack.io

Helm install (make install-rag) -- per-namespace
  |
  +-- rag chart (deploy/helm/rag/)
       |
       +-- llama-stack-instance (new)    <-- local subchart at rag/llama-stack-instance/
       |     Referenced via file://llama-stack-instance in Chart.yaml
       |     Creates: LlamaStackDistribution CR, ConfigMap, PVC, Secret
       |     The RHOAI-managed OPERATOR reconciles the CR into: Deployment, Service
       |
       +-- llm-service v0.5.4           <-- unchanged
       |
       +-- pgvector v0.5.0              <-- unchanged
```

**How the Deployment gets created:** A common point of confusion is that you will still see a `llamastack` Deployment and `llamastack-service` Service in the application namespace after migration. These are **not** leftover resources from the old Helm-based approach -- they are created and managed by the RHOAI LlamaStack Operator. The flow is:

1. `make install-rag` deploys the RAG Helm chart, which includes the `llama-stack-instance` subchart
2. The subchart renders a `LlamaStackDistribution` CR (not a Deployment) and applies it to the namespace
3. The RHOAI LlamaStack Operator (running in `redhat-ods-applications`) watches for `LlamaStackDistribution` CRs across all namespaces
4. When it detects the CR, the operator reconciles it into a Deployment + Service in the same namespace as the CR
5. The operator continuously manages these resources -- if the Deployment is deleted, the operator recreates it; if the CR is updated, the operator rolls out the changes

The key difference from the old approach: Helm previously owned the Deployment directly. Now, Helm only owns the CR, and the operator owns the Deployment. This gives you operator-managed lifecycle (health checks, automatic reconciliation, status reporting via CR conditions) instead of static Helm-managed resources.

### Options A/B: Standalone operator (alternative)

```
Phase 1: Install operator (one-time, cluster-scoped)
  |
  +-- llama-stack-operator chart (deploy/helm/llama-stack-operator/)
       Creates: CRD, operator Deployment, RBAC, ServiceAccount
       Namespace: llama-stack-k8s-operator-system
       Image: quay.io/eformat/llama-stack-k8s-operator:v0.3.0

Phase 2: Install RAG (per-namespace)
  |
  +-- rag chart (deploy/helm/rag/)
       |
       +-- llama-stack-instance (new)    <-- replaces llama-stack dependency
       |     Creates: LlamaStackDistribution CR, ConfigMap, PVC
       |     The OPERATOR then reconciles the CR into: Deployment, Service
       |
       +-- llm-service v0.5.4           <-- unchanged
       |
       +-- pgvector v0.5.0              <-- unchanged
```

---

## What Does NOT Change

These components are unaffected because they consume LlamaStack via `LLAMA_STACK_URL` (an HTTP endpoint), which remains the same:

| Component | Why unchanged |
|-----------|--------------|
| `src/core/config.py` | Reads `LLAMA_STACK_URL` env var, no deployment awareness |
| `src/chatbots/llama_bot.py` | Uses OpenAI SDK pointed at `LLAMA_STACK_URL` |
| `src/core/llm_client.py` | Uses `LLAMA_STACK_URL` for chat completions |
| `src/alerting/alert_receiver.py` | Uses `LLAMA_STACK_URL` from config |
| `src/mcp_server/setup_integration.py` | Provides default `LLAMA_STACK_URL` for Claude Desktop |
| `deploy/helm/mcp-server/` | Only the `LLAMA_STACK_URL` value may need updating (see Phase 5) |
| `deploy/helm/alerting/` | Only the `LLAMA_STACK_URL` value may need updating (see Phase 5) |
| `llm-service` chart | Deploys vLLM independently, LlamaStack points to it |
| `pgvector` chart | Deploys PostgreSQL independently, LlamaStack connects to it |
| React UI / Console Plugin | Talks to MCP server, not LlamaStack directly |
| All Python tests | Mock LlamaStack interactions, no deployment awareness |

---

## Migration Steps

### Phase 1: ~~Add the LlamaStack Operator Chart~~ N/A (RHOAI-managed)

> **NOT APPLICABLE.** We are using Option C (RHOAI-managed LlamaStack Operator). The operator is installed and managed by RHOAI as part of the `DataScienceCluster` configuration — no separate operator chart is needed. See [Cluster Preparation](#cluster-preparation-rhoai--prerequisite-operators) for the one-time RHOAI setup that was completed on 2026-03-23.
>
> The `check-llamastack-operator` Makefile target (added in Phase 4) verifies the CRD is present before deploying the RAG chart.

---

### Phase 2: Create LlamaStack Instance Chart (Replace llama-stack Dependency)

**Goal:** Replace the ai-architecture-charts `llama-stack` Helm chart with a new local subchart that creates a `LlamaStackDistribution` CR instead of a raw Deployment.

**Action:** Create `deploy/helm/rag/llama-stack-instance/` (local subchart alongside the rag chart's `charts/` directory, not inside it — `charts/` is gitignored and generated by `helm dependency update`). The rag chart references it via `repository: "file://llama-stack-instance"` in `Chart.yaml`, and `helm dependency update` packages it into `charts/llama-stack-instance-1.0.0.tgz`.

**Files to create:**

```
deploy/helm/rag/llama-stack-instance/
  Chart.yaml
  values.yaml
  templates/
    _helpers.tpl
    llamastackdistribution.yaml    # The CR that the operator reconciles
    configmap.yaml                  # LlamaStack run config (config.yaml)
    pvc.yaml                        # Persistent storage
    secret.yaml                     # Env secrets (e.g., TAVILY_SEARCH_API_KEY)
```

**The `LlamaStackDistribution` CR template** (`llamastackdistribution.yaml`):

```yaml
apiVersion: llamastack.io/v1alpha1
kind: LlamaStackDistribution
metadata:
  name: llamastack                   # <-- MUST match current service name for compatibility
spec:
  replicas: {{ .Values.replicas | default 1 }}
  server:
    distribution:
      image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
    containerSpec:
      name: llama-stack
      port: 8321
      env:
        # Model, OTEL, pgvector env vars (migrated from current rag/values.yaml)
      resources: {{ .Values.resources }}
    userConfig:
      configMapName: llamastack-config
    podOverrides:
      volumeMounts:
        - name: dot-llama
          mountPath: /.llama
        - name: cache
          mountPath: /.cache
      volumes:
        - name: dot-llama
          persistentVolumeClaim:
            claimName: llama-stack-data
        - name: cache
          emptyDir: {}
```

**The ConfigMap template** (`configmap.yaml`):

This is the critical piece. The current ai-architecture-charts `llama-stack` chart generates a `config.yaml` (version 2) with dynamic model registration via `global.models`. The new template needs to preserve this flexibility.

Two approaches:

- **Option A (simpler):** Port the existing `configmap.yaml` template logic from ai-architecture-charts into the new subchart. This preserves the `global.models` pattern and the `mergeModels` helper.
- **Option B (lls-observability style):** Use a static `run.yaml` with environment variable substitution. Simpler template but less flexible for multi-model scenarios.

**Recommendation:** Option A -- port the existing configmap template. The `global.models` pattern is already used throughout the Makefile and is the expected interface for users deploying with `make install LLM=...`.

**Key config mapping:**

| Current (ai-architecture-charts) | New (instance CR) |
|----------------------------------|-------------------|
| `llama-stack.image` | `spec.server.distribution.image` |
| `llama-stack.env` | `spec.server.containerSpec.env` |
| `llama-stack.resources` | `spec.server.containerSpec.resources` |
| `llama-stack.service.port` (8321) | `spec.server.containerSpec.port` (8321) |
| `run-config` ConfigMap | `spec.server.userConfig.configMapName` |
| `llama-stack.volumes/volumeMounts` | `spec.server.podOverrides.volumes/volumeMounts` |
| `global.models` (mergeModels helper) | Same helper, ported to new chart |

---

### Phase 3: Update RAG Chart to Use the New Instance Chart

**Goal:** Swap the dependency in the rag chart.

**File: `deploy/helm/rag/Chart.yaml`**

Remove:
```yaml
dependencies:
  - name: llama-stack
    version: 0.5.3
    repository: https://rh-ai-quickstart.github.io/ai-architecture-charts
```

Add (local subchart via `file://` reference):
```yaml
dependencies:
  - name: llama-stack-instance
    version: 1.0.0
    repository: "file://llama-stack-instance"
```

**File: `deploy/helm/rag/values.yaml`**

Rename the `llama-stack:` values section to `llama-stack-instance:` and update the schema to match the new chart's values.yaml. The env vars, MCP server config, and pgvector references should carry over with minimal changes.

**File: `deploy/helm/rag/Chart.lock`**

Delete and regenerate via `helm dependency update`.

**Cleanup:** Remove `deploy/helm/rag/charts/llama-stack-0.5.3.tgz` (the cached dependency archive).

---

### Phase 4: Update Makefile

**Goal:** Add operator verification (or installation for standalone), update helm args for the new chart structure.

#### 4a. New target: `check-llamastack-operator` / `install-operator`

For RHOAI deployments (Option C), the operator is already running. The target should **verify** CRD presence rather than install:

```makefile
.PHONY: check-llamastack-operator
check-llamastack-operator:
	@echo "Checking LlamaStack Operator CRD..."
	@oc get crd llamastackdistributions.llamastack.io > /dev/null 2>&1 || \
		{ echo "ERROR: LlamaStackDistribution CRD not found. Ensure RHOAI is installed with llamastackoperator: Managed in the DataScienceCluster."; exit 1; }
	@echo "LlamaStack Operator CRD is registered."
```

For standalone deployments (Options A/B only), keep the Helm-based install:

```makefile
.PHONY: install-operator
install-operator:
	@echo "Installing LlamaStack Operator..."
	@cd deploy/helm && helm upgrade --install llama-stack-operator llama-stack-operator \
		--timeout 5m --wait
	@echo "Waiting for CRD to be registered..."
	@kubectl wait --for=condition=Established crd/llamastackdistributions.llamastack.io --timeout=60s
```

#### 4b. Update `helm_llama_stack_args`

Current (lines 139-147):
```makefile
helm_llama_stack_args = \
    $(if $(LLM),--set global.models.$(LLM).enabled=true,) \
    ...
    $(if $(LLAMA_STACK_ENV),--set-json llama-stack.secrets='$(LLAMA_STACK_ENV)',) \
    $(if $(RAW_DEPLOYMENT),--set llama-stack.rawDeploymentMode=$(RAW_DEPLOYMENT),)
```

Updated (key prefix changes from `llama-stack` to `llama-stack-instance`):
```makefile
helm_llama_stack_args = \
    $(if $(LLM),--set global.models.$(LLM).enabled=true,) \
    ...
    $(if $(LLAMA_STACK_ENV),--set-json llama-stack-instance.secrets='$(LLAMA_STACK_ENV)',) \
```

Note: The `global.models` pattern does not change -- it's a Helm globals mechanism that works across any subchart that reads `$.Values.global.models`.

#### 4c. Update `install-rag` to verify operator is available

The operator CRD must exist before the rag chart can create `LlamaStackDistribution` CRs:

- **RHOAI (Option C):** Make `install-rag` depend on `check-llamastack-operator`: `install-rag: namespace check-llamastack-operator`
- **Standalone (Options A/B):** Make `install-rag` depend on `install-operator`: `install-rag: namespace install-operator`

#### 4d. New target: `uninstall-operator`

```makefile
.PHONY: uninstall-operator
uninstall-operator:
	@echo "Uninstalling LlamaStack Operator..."
	@cd deploy/helm && helm uninstall llama-stack-operator || true
```

---

### Phase 5: Update Service References

**Goal:** Ensure all components can still reach LlamaStack after migration.

The operator creates a Kubernetes Service named after the `LlamaStackDistribution` CR's `metadata.name`. If we name the CR `llamastack`, the operator will create a Service called `llamastack` on port 8321 -- **matching the current convention exactly**.

**Verification needed:** Confirm that the operator-created Service uses port 8321 (not a different port like 80). The lls-observability setup accesses the instance at port 80 (`http://llama-stack-instance.llama-serve.svc.cluster.local:80`), which suggests the operator may use port 80 by default. If so, we need either:

- Configure the CR to use port 8321, or
- Update all URL references to use the operator's default port

**Files to check/update if the port changes:**

| File | Current URL | What to update |
|------|-------------|----------------|
| `deploy/helm/mcp-server/templates/deployment.yaml:80` | `http://llamastack.{ns}.svc.cluster.local:8321/v1/openai/v1` | Port number |
| `deploy/helm/alerting/templates/cronjob.yaml:25` | `http://llamastack.{ns}.svc.cluster.local:8321` | Port number |
| `scripts/local-dev.sh:20` | `LLAMASTACK_SERVICE_PORT=8321` | Port number |
| `src/core/config.py:84` | Default `http://localhost:8321/v1/openai/v1` | Port number (local dev) |

**If the CR name is `llamastack` and port is 8321, no changes are needed in these files.**

---

### Phase 6: Update Local Development

**Goal:** Ensure `scripts/local-dev.sh` still works.

**File: `scripts/local-dev.sh`**

The script finds the LlamaStack service by label:
```bash
LLAMASTACK_SERVICE=$(oc get services -n "$LLAMA_MODEL_NAMESPACE" -o name -l "$LLAMASTACK_SERVICE_LABEL")
```

The operator-created Service will have different labels than the current Helm-created Service. The label selector in `local-dev.sh` needs to be updated to match whatever labels the operator assigns.

**Action:** After deploying the operator locally, check the labels on the created Service and update `LLAMASTACK_SERVICE_LABEL` accordingly. Alternatively, switch to selecting by service name (`llamastack`) instead of label.

---

### Phase 7: Testing and Validation

#### 7a. Helm template validation

```bash
cd deploy/helm
helm template test-release rag \
  --set global.models.llama-3-1-8b-instruct.enabled=true \
  --set llm-service.secret.hf_token=test \
  | grep -A 20 'kind: LlamaStackDistribution'
```

Verify the rendered CR has the correct:
- `metadata.name: llamastack`
- `spec.server.containerSpec.port: 8321`
- `spec.server.userConfig.configMapName` pointing to the right ConfigMap
- Environment variables matching the current deployment

#### 7b. Deployment test (on a dev cluster)

1. `make install-operator`
2. Verify CRD is registered: `kubectl get crd llamastackdistributions.llamastack.io`
3. `make install-rag LLM=llama-3-1-8b-instruct NAMESPACE=test-migration`
4. Verify CR is created: `kubectl get llamastackdistributions -n test-migration`
5. Verify CR reaches `Ready` phase: `kubectl get llsd -n test-migration -o wide`
6. Verify Service exists: `kubectl get svc llamastack -n test-migration`
7. Verify MCP server can reach LlamaStack: `kubectl logs deploy/aiobs-mcp-server -n test-migration`
8. End-to-end: use the console plugin or React UI to ask a question and verify LLM response

#### 7c. Local dev test

1. Deploy operator + rag chart on cluster
2. Run `scripts/local-dev.sh`
3. Verify LlamaStack port-forward succeeds
4. Verify `curl http://localhost:8321/v1/openai/v1/models` returns models

#### 7d. Automated tests

- Run `make test` -- all existing Python and React tests should pass without changes (they mock LlamaStack)

---

## Key Decisions

### Decision 1: Operator chart source -- DECIDED: Option C

| Option | Pros | Cons |
|--------|------|------|
| A: Copy from lls-observability | Full control, can customize | Must manually sync updates |
| B: Remote Helm dependency | Auto-updates, single source of truth | Requires lls-observability to publish to a chart repo |
| **C: Use RHOAI-managed operator** | **Zero maintenance, platform-supported** | **Requires RHOAI to be installed** |

**Chosen: Option C.** RHOAI 3.3 installed on the cluster with `llamastackoperator: Managed` in the DataScienceCluster. See [Cluster Preparation](#cluster-preparation-rhoai--prerequisite-operators).

### Decision 2: LlamaStack image -- DECIDED: Option A

| Option | Image | Notes |
|--------|-------|-------|
| **A: Keep current** | **`llamastack/distribution-starter:0.6.0`** | **Known working, ai-architecture-charts config format** |
| B: Use lls-observability image | `quay.io/rhoai-genaiops/llama-stack-vllm-milvus:0.2.11` | Older version, Milvus-specific |
| C: Use latest upstream | `llamastack/distribution-starter:latest` | Newest features but needs testing |

**Chosen: Option A.** Keep the current image to minimize migration variables. Upgrade separately once the operator deployment is validated.

### Decision 3: ConfigMap format -- DECIDED: Option A

| Option | Format | Notes |
|--------|--------|-------|
| **A: Port existing template** | **`config.yaml` (version 2)** | **Preserves `global.models` pattern, `mergeModels` helper** |
| B: Use lls-observability format | `run.yaml` (version 1) | Simpler but requires reworking model configuration |

**Chosen: Option A.** The `global.models` merge pattern is deeply integrated into the Makefile and provides the `make install LLM=... DEVICE=...` user experience.

### Decision 4: Port number and Service name -- DECIDED

Verified on 2026-03-23 by deploying a test `LlamaStackDistribution` CR:

| Property | Operator behavior |
|---|---|
| **Service name** | `{cr-name}-service` (e.g., CR named `llamastack` -> Service `llamastack-service`) |
| **Service port** | Uses `spec.server.containerSpec.port` from the CR (set to `8321`, confirmed working) |
| **Service selector** | `app: llama-stack`, `app.kubernetes.io/instance: {cr-name}` |
| **Service URL pattern** | `http://{cr-name}-service.{namespace}.svc.cluster.local:{port}` |

**Chosen:** Accept the operator's `-service` suffix naming and update all `LLAMA_STACK_URL` references from `llamastack` to `llamastack-service` (Phase 5). Port 8321 is preserved -- no port changes needed.

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Operator-created Service has different name/port | High -- breaks all consumers | Medium | Name CR `llamastack`, verify port in CR spec |
| Operator-created Service has different labels | Low -- only affects local-dev.sh | High | Update label selector or use name-based selection |
| ConfigMap format incompatibility | High -- LlamaStack fails to start | Low | Keep existing config.yaml format (Decision 3, Option A) |
| Operator not ready when CR is applied | Medium -- CR stuck in Pending | Medium | Add `kubectl wait` for CRD in Makefile |
| Operator version drift from lls-observability | Low | Medium | Pin version, document upgrade process |
| `global.models` merge pattern breaks with new chart | High -- install commands stop working | Low | Port the `mergeModels` helper into new subchart |

---

## Design Validation: Comparison with lls-observability

The `rh-ai-quickstart/lls-observability` repository is the reference implementation for the operator-based LlamaStack deployment. After reviewing their implementation (Helm charts, CR templates, ConfigMap, install scripts), the table below summarizes where our design aligns and where it intentionally diverges.

### Source repository

- **Repo:** `rh-ai-quickstart/lls-observability`
- **Key paths examined:**
  - `helm/01-operators/llama-stack-operator/` -- standalone operator chart (CRD, RBAC, controller Deployment)
  - `helm/03-ai-services/llama-stack-instance/` -- instance chart (LlamaStackDistribution CR, ConfigMap with `run.yaml`, PVC, OTel sidecar)
  - `helm/03-ai-services/llama3.2-3b/` -- KServe InferenceService + ServingRuntime for vLLM model serving
  - `scripts/install-full-stack.sh` -- phased deployment script

### How lls-observability deploys LlamaStack

1. **Operator install:** Standalone Helm chart creates the CRD, RBAC, and controller-manager Deployment (image: `quay.io/eformat/llama-stack-k8s-operator:v0.3.0`) in namespace `llama-stack-k8s-operator-system`
2. **Instance chart creates:**
   - A `LlamaStackDistribution` CR with env vars flat in `spec.server.containerSpec.env`
   - A ConfigMap (`llama-stack-config`) containing `run.yaml` (v1 format) referenced via `spec.server.userConfig.configMapName`
   - A PVC (`llama-stack-persist`, 5Gi) -- created by the Helm chart, not the operator
   - An `OpenTelemetryCollector` sidecar CR for trace export
3. **vLLM connection:** LlamaStack connects directly to the KServe predictor service at `http://llama3-2-3b-predictor:8080/v1`, set both as env var (`VLLM_URL`) and in the `run.yaml` provider config (`remote::vllm`)
4. **Service port:** Their playground connects at port `80`, suggesting the operator-created Service maps port 80 -> container port 8321

### Comparison

| Decision | lls-observability | Our branch | Verdict |
|----------|------------------|-------------|---------|
| Operator source | Standalone Helm chart (`quay.io/eformat/...`) | RHOAI-managed (`DataScienceCluster`) | Ours is better -- zero operator maintenance, platform-supported |
| Config format | `run.yaml` v1 | `config.yaml` v2 with `global.models` helper | Ours preserves the `make install LLM=...` UX |
| LlamaStack image | `quay.io/rhoai-genaiops/llama-stack-vllm-milvus:0.2.11` | `llamastack/distribution-starter:0.6.0` | Ours is newer |
| Vector store | Milvus (embedded SQLite-backed) | pgvector (standalone PostgreSQL) | Both valid; ours matches existing infrastructure |
| PVC | Chart-created, 5Gi | Chart-created | Same pattern |
| CR structure | Env vars flat in `containerSpec.env` | Same | Aligned |
| ConfigMap reference | `userConfig.configMapName` | Same | Aligned |
| NetworkPolicy | Not addressed | Additive policy for same-namespace access | We identified and solved a real operator issue (Issue 3) |
| OTel integration | Sidecar `OpenTelemetryCollector` CR | Env vars pointing to cluster OTel collector | Different mechanism, same outcome |

### Conclusion

The structural patterns match -- CR-based instance, chart-created PVC, ConfigMap for runtime config, env vars for vLLM connection. The divergences (RHOAI-managed operator, config.yaml v2, newer image, pgvector, NetworkPolicy fix) are all intentional and justified.

**Watch item:** Service port. lls-observability's Service uses port `80` (mapped to container `8321`). Our cluster testing confirmed that `spec.server.containerSpec.port: 8321` works, but verify the actual operator-created Service port after deployment and update `LLAMA_STACK_URL` references if needed (Phase 5).

---

## Reference: lls-observability Operator Model

### CRD Schema (`llamastack.io/v1alpha1`, kind: `LlamaStackDistribution`)

```yaml
spec:
  replicas: int32
  server:
    distribution:
      name: string        # Named distribution (mutually exclusive with image)
      image: string       # Direct image reference
    containerSpec:
      name: string
      port: int32
      env: []EnvVar
      args: []string
      command: []string
      resources: ResourceRequirements
    userConfig:
      configMapName: string
      configMapNamespace: string
    podOverrides:
      serviceAccountName: string
      volumeMounts: []VolumeMount
      volumes: []Volume
status:
  phase: Pending | Initializing | Ready | Failed | Terminating
  availableReplicas: int32
  conditions: []Condition
```

### Operator image

- **Standalone (lls-observability, Options A/B):** `quay.io/eformat/llama-stack-k8s-operator:v0.3.0`
- **RHOAI-managed (Option C):** Image is managed by the RHOAI operator and is not user-configurable. The version is determined by the RHOAI release channel (e.g., `stable-3.3`).

### Install order (from lls-observability)

1. Operator (Phase 2 in their install script)
2. Wait for CRD registration
3. AI services including LlamaStackDistribution CR (Phase 6)
