{{/*
Expand the name of the chart.
*/}}
{{- define "llama-stack-instance.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "llama-stack-instance.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "llama-stack-instance.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "llama-stack-instance.labels" -}}
helm.sh/chart: {{ include "llama-stack-instance.chart" . }}
{{ include "llama-stack-instance.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "llama-stack-instance.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llama-stack-instance.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "llama-stack-instance.mergeModels" -}}
  {{- $root := . }}
  {{- $globalModels := .Values.global | default dict }}
  {{- $globalModels := $globalModels.models | default dict }}
  {{- $localModels := .Values.models | default dict }}
  {{- $merged := merge $globalModels $localModels }}
  {{- range $key, $model := $merged }}
    {{- if not $model.url }}
      {{- $url := printf "https://%s.%s.svc.cluster.local/v1" $key $root.Release.Namespace }}
      {{- if $root.Values.rawDeploymentMode }}
        {{- $url = printf "http://%s-vllm.%s.svc.cluster.local/v1" $key $root.Release.Namespace }}
      {{- end }}
      {{- $_ := set $merged $key (merge $model (dict "url" $url)) }}
    {{- end }}
  {{- end }}
  {{- toJson $merged }}
{{- end }}

{{- define "llama-stack-instance.mergeMcpServers" -}}
  {{- $globalServers := .Values.global | default dict }}
  {{- $globalServers := index $globalServers "mcp-servers" | default dict }}
  {{- $localServers := index .Values "mcp-servers" | default dict }}
  {{- $merged := merge $globalServers $localServers }}
  {{- toJson $merged }}
{{- end }}

{{/*
LlamaStack server configuration content.
Emitted under both 'config.yaml' (for operator v0.4.0+ / RHOAI 3.x) and
'run.yaml' (for operator v0.3.0 / RHOAI 2.x) in the ConfigMap.
*/}}
{{- define "llama-stack-instance.configContent" -}}
{{- $models := include "llama-stack-instance.mergeModels" . | fromJson -}}
{{- $mcpServers := include "llama-stack-instance.mergeMcpServers" . | fromJson -}}
{{- $shields := list -}}
{{- range $key, $model := $models -}}
  {{- if and $model.enabled $model.registerShield -}}
    {{- $shields = append $shields $model -}}
  {{- end -}}
{{- end -}}
version: '2'
image_name: remote-vllm
apis:
- agents
- datasetio
- eval
- files
- inference
- safety
- scoring
- telemetry
- tool_runtime
- vector_io
providers:
  inference:
  {{- range $key, $model := $models }}
  {{- if $model.enabled }}
  - provider_id: {{ $key }}
    provider_type: remote::vllm
    config:
      url: {{ $model.url }}
      api_token: {{ $model.apiToken | default "fake" }}
      max_tokens: {{ $model.maxTokens | default 4096 }}
      tls_verify: {{ $model.tlsVerify | default false }}
  {{- end }}
  {{- end }}
  {{- if .Values.vertexai.enabled }}
  - provider_id: {{ .Values.vertexai.projectId }}-vertexai
    provider_type: remote::vertexai
    config:
      project: {{ .Values.vertexai.projectId }}
      location: {{ .Values.vertexai.location }}
  {{- end }}
  - provider_id: sentence-transformers
    provider_type: inline::sentence-transformers
    config: {}
  vector_io:
  - provider_id: pgvector
    provider_type: remote::pgvector
    config:
      host: ${env.POSTGRES_HOST:=pgvector}
      port: ${env.POSTGRES_PORT:=5432}
      db: ${env.POSTGRES_DBNAME:=rag_blueprint}
      user: ${env.POSTGRES_USER:=postgres}
      password: ${env.POSTGRES_PASSWORD:=rag_password}
      kvstore:
        {{- toYaml .Values.vectorIOKvstore | nindent 8 }}
  files:
  - provider_id: meta-reference-files
    provider_type: inline::localfs
    config:
      storage_dir: ${env.FILES_STORAGE_DIR:=/tmp/llama-stack-files}
      metadata_store:
        type: sqlite
        db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/starter}/files_metadata.db
  safety:
  - provider_id: llama-guard
    provider_type: inline::llama-guard
    config: {}
  agents:
  {{- with .Values.providers.agents }}
  {{- toYaml . | nindent 2 }}
  {{- end }}
  eval:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      kvstore:
        type: sqlite
        namespace: null
        db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/starter}/meta_reference_eval.db
  datasetio:
  - provider_id: huggingface
    provider_type: remote::huggingface
    config:
      kvstore:
        type: sqlite
        namespace: null
        db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/starter}/huggingface_datasetio.db
  - provider_id: localfs
    provider_type: inline::localfs
    config:
      kvstore:
        type: sqlite
        namespace: null
        db_path: ${env.SQLITE_STORE_DIR:=~/.llama/distributions/starter}/localfs_datasetio.db
  scoring:
  - provider_id: basic
    provider_type: inline::basic
    config: {}
  - provider_id: llm-as-judge
    provider_type: inline::llm-as-judge
    config: {}
  - provider_id: braintrust
    provider_type: inline::braintrust
    config:
      openai_api_key: ${env.OPENAI_API_KEY:=}
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      sqlite_db_path: ${env.SQLITE_DB_PATH:=~/.llama/distributions/starter/trace_store.db}
      {{- if .Values.otelExporter }}
      sinks: ${env.TELEMETRY_SINKS:=console,sqlite,otel_trace}
      service_name: {{ include "llama-stack-instance.fullname" . }}
      otel_exporter_otlp_endpoint: ${env.OTEL_EXPORTER_OTLP_ENDPOINT:=}
      {{- else }}
      sinks: ${env.TELEMETRY_SINKS:=console,sqlite}
      {{- end }}
  tool_runtime:
  - provider_id: brave-search
    provider_type: remote::brave-search
    config:
      api_key: ${env.BRAVE_SEARCH_API_KEY:=}
      max_results: 3
  - provider_id: tavily-search
    provider_type: remote::tavily-search
    config:
      api_key: ${env.TAVILY_SEARCH_API_KEY:=}
      max_results: 3
  - provider_id: rag-runtime
    provider_type: inline::rag-runtime
    config: {}
  - provider_id: model-context-protocol
    provider_type: remote::model-context-protocol
    config: {}
metadata_store:
  {{- toYaml .Values.metadataStore | nindent 2 }}
models:
{{- range $key, $model := $models }}
{{- if $model.enabled }}
- metadata: {}
  model_id: {{ $model.id }}
  provider_id: {{ $key }}
  model_type: llm
{{- end }}
{{- end }}
- metadata:
    embedding_dimension: 384
  model_id: all-MiniLM-L6-v2
  provider_id: sentence-transformers
  model_type: embedding
{{- with $shields }}
shields:
{{- range $model := . }}
- shield_id: {{ $model.id }}
  provider_id: llama-guard
{{- end }}
{{- end }}
vector_dbs: []
datasets: []
scoring_fns: []
eval_tasks: []
tool_groups:
- toolgroup_id: builtin::websearch
  provider_id: tavily-search
- toolgroup_id: builtin::rag
  provider_id: rag-runtime
{{- range $key, $server := $mcpServers }}
- toolgroup_id: mcp::{{ $key }}
  provider_id: model-context-protocol
  mcp_endpoint:
    uri: {{ $server.uri }}
{{- end }}
{{- if .Values.auth }}
server:
  auth:
    {{- with .Values.auth.provider_config }}
    provider_config:
      {{- toYaml . | nindent 6 }}
    {{- end }}
    {{- with .Values.auth.access_policy }}
    access_policy:
      {{- toYaml . | nindent 4 }}
    {{- end }}
{{- end }}
{{ end }}
