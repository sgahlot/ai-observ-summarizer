#!/bin/bash

# AI Observability Metric Summarizer - Local Development Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_NAMESPACE="openshift-monitoring"
OBSERVABILITY_NAMESPACE="observability-hub"
METRIC_API_APP="metrics-api-app"
THANOS_PORT=9090
TEMPO_PORT=8082
LLAMASTACK_PORT=8321
LLAMA_MODEL_PORT=8080
# Metrics API (FastAPI) port for local dev; can override via METRICS_API_PORT
METRICS_API_PORT=${METRICS_API_PORT:-8000}
UI_PORT=8501
MCP_PORT=${MCP_PORT:-8085}

echo -e "${BLUE}🚀 AI Observability Metric Summarizer - Local Development Setup${NC}"
echo "=============================================================="

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n/-N NAMESPACE              Default namespace for pods (required)"
    echo "  -m/-M NAMESPACE              Llama Model namespace (optional, use if model is in different namespace)"
    echo "  -c/-C CONFIG                 Model config source: 'local' or 'cluster' (default: local)"
    echo "  -t/-T TRANSPORT              MCP transport protocol: 'http', 'streamable-http', 'sse' (default: http)"
    echo ""
    echo "Examples:"
    echo "  $0 -n default-ns                       # All pods/services in same namespace, use local config, http transport"
    echo "  $0 -N default-ns                       # All pods/services in same namespace (uppercase), use local config"
    echo "  $0 -n default-ns -m model-ns           # Model in different namespace, use local config"
    echo "  $0 -n default-ns -c cluster            # Use cluster model config instead of local"
    echo "  $0 -n default-ns -C local              # Explicitly use local model config (default)"
    echo "  $0 -n default-ns -t streamable-http    # Use streamable-http transport for MCP Inspector"
    echo "  $0 -n default-ns -T SSE                # Use SSE transport (uppercase)"
}

# Function to parse command line arguments
parse_args() {
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        usage
        exit 2
    fi

    DEFAULT_NAMESPACE=""
    LLAMA_MODEL_NAMESPACE=""
    MODEL_CONFIG_SOURCE="local"  # Default to local
    MCP_TRANSPORT_PROTOCOL="http"  # Default to http

    # Parse standard arguments using getopts
    while getopts "n:N:m:M:c:C:t:T:" opt; do
        case $opt in
            n|N) DEFAULT_NAMESPACE="$OPTARG"
                 ;;
            m|M) LLAMA_MODEL_NAMESPACE="$OPTARG"
                 ;;
            c|C) MODEL_CONFIG_SOURCE="$OPTARG"
                 ;;
            t|T) MCP_TRANSPORT_PROTOCOL="$OPTARG"
                 ;;
            *) echo -e "${RED}❌ INVALID option: [$OPTARG]${NC}"
               usage
               exit 1
               ;;
        esac
    done

    # Validate arguments
    if [ -z "$DEFAULT_NAMESPACE" ]; then
        echo -e "${RED}❌ Default namespace is required. Please specify using -n or -N${NC}"
        usage
        exit 1
    fi

    # Validate model config source
    if [[ "$MODEL_CONFIG_SOURCE" != "local" && "$MODEL_CONFIG_SOURCE" != "cluster" ]]; then
        echo -e "${RED}❌ Invalid model config source: $MODEL_CONFIG_SOURCE${NC}"
        echo -e "${YELLOW}   Valid options: 'local' or 'cluster'${NC}"
        usage
        exit 1
    fi

    # Validate transport protocol
    if [[ "$MCP_TRANSPORT_PROTOCOL" != "http" && "$MCP_TRANSPORT_PROTOCOL" != "streamable-http" && "$MCP_TRANSPORT_PROTOCOL" != "sse" ]]; then
        echo -e "${RED}❌ Invalid MCP transport protocol: $MCP_TRANSPORT_PROTOCOL${NC}"
        echo -e "${YELLOW}   Valid options: http, streamable-http, sse${NC}"
        usage
        exit 1
    fi

    # Set llama model namespace to default if not provided
    if [ -z "$LLAMA_MODEL_NAMESPACE" ]; then
        LLAMA_MODEL_NAMESPACE="$DEFAULT_NAMESPACE"
    fi
}

# Function to cleanup on exit
cleanup() {
    # Prevent multiple cleanup calls
    if [ "$CLEANUP_DONE" = "true" ]; then
        return
    fi
    CLEANUP_DONE=true

    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"

    # Kill background monitoring
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi

    # Kill OpenShift port-forwards
    kill_port_forwards false
    
    # Kill local services
    pkill -f "uvicorn.*metrics_api:app" >/dev/null 2>&1 || true
    pkill -f "mcp_server.main" >/dev/null 2>&1 || true
    pkill -f "streamlit run ui.py" >/dev/null 2>&1 || true

    # Deactivate virtual environment if it was activated
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${BLUE}🐍 Deactivating virtual environment...${NC}"
        deactivate
    fi

    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Function to check prerequisites and activate virtual environment
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

    # Check for virtual environment and activate it
    if [ -f ".venv/bin/activate" ]; then
        echo -e "${BLUE}🐍 Activating Python virtual environment...${NC}"
        source .venv/bin/activate
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${YELLOW}⚠️  Virtual environment (.venv) not found${NC}"
        echo -e "${YELLOW}   Please create virtual environment by following README or DEV_GUIDE${NC}"
        exit 1
    fi

    if ! command -v oc &> /dev/null; then
        echo -e "${RED}❌ OpenShift CLI (oc) is not installed${NC}"
        exit 1
    fi

    if ! oc whoami &> /dev/null; then
        echo -e "${RED}❌ Not logged in to OpenShift cluster${NC}"
        echo -e "${YELLOW}   Please run: oc login${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Helper function to create port-forward command
create_port_forward() {
    local resource_type="$1"  # "pod" or "service"
    local resource_name="$2"
    local local_port="$3"
    local remote_port="$4"
    local namespace="$5"
    local description="$6"
    
    oc port-forward $resource_type/"$resource_name" "$local_port:$remote_port" -n "$namespace" >/dev/null 2>&1 &
    echo -e "${GREEN}✅ Found $description: 📊 ($resource_type: $resource_name, ns: $namespace) available at: http://localhost:$local_port${NC}"
}

# Helper function to check port-forward status
check_port_status() {
    local port="$1"
    local service_name="$2"
    local resource_info="$3"
    
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $service_name port-forward (port $port → $resource_info) is active${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name port-forward (port $port → $resource_info) is NOT active${NC}"
        return 1
    fi
}

# Helper function to kill port-forward processes
kill_port_forwards() {
    local suppress_output="${1:-true}"  # Default to true for backward compatibility
    
    if [ "$suppress_output" = "true" ]; then
        # Full suppression for background operations
        (pkill -f "oc port-forward" >/dev/null 2>&1 || true) 2>/dev/null
    else
        # Basic suppression for cleanup operations
        pkill -f "oc port-forward" >/dev/null 2>&1 || true
    fi
}

# Function to find and start port forwards
start_port_forwards() {
    echo -e "${BLUE}🔍 Finding pods and starting port-forwards...${NC}"
    
    # Kill any existing port-forwards to prevent duplicates
    echo -e "${YELLOW}🧹 Cleaning up existing port-forwards...${NC}"
    kill_port_forwards true
    sleep 1
    
    # Find Thanos service (prefer service over pod for better reliability)
    THANOS_SERVICE=$(oc get services -n "$PROMETHEUS_NAMESPACE" -o name | grep thanos-querier | head -1 | cut -d'/' -f2 || echo "")
    if [ -z "$THANOS_SERVICE" ]; then
        # Fallback to pod if service not found
        THANOS_POD=$(oc get pods -n "$PROMETHEUS_NAMESPACE" -o name | grep thanos-querier | head -1 | cut -d'/' -f2 || echo "")
        if [ -n "$THANOS_POD" ]; then
            create_port_forward "pod" "$THANOS_POD" "$THANOS_PORT" "9091" "$PROMETHEUS_NAMESPACE" "Thanos"
        else
            echo -e "${RED}❌ No Thanos service or pod found${NC}"
            exit 1
        fi
    else
        create_port_forward "service" "$THANOS_SERVICE" "$THANOS_PORT" "9091" "$PROMETHEUS_NAMESPACE" "Thanos"
    fi
    
    # Find LlamaStack pod
    LLAMASTACK_POD=$(oc get pods -n "$DEFAULT_NAMESPACE" -o name | grep -E "(llama-stack|llamastack)" | head -1 | cut -d'/' -f2 || echo "")
    if [ -n "$LLAMASTACK_POD" ]; then
        create_port_forward "pod" "$LLAMASTACK_POD" "$LLAMASTACK_PORT" "8321" "$DEFAULT_NAMESPACE" "LlamaStack"
    else
        echo -e "${RED}❌  LlamaStack pod not found. Exiting...${NC}"
        exit 1
    fi
    
    # Find Llama Model service
    LLAMA_MODEL_SERVICE=$(oc get services -n "$LLAMA_MODEL_NAMESPACE" -o name | grep -E "(llama-3|predictor)" | grep -v stack | head -1 | cut -d'/' -f2 || echo "")
    if [ -n "$LLAMA_MODEL_SERVICE" ]; then
        create_port_forward "service" "$LLAMA_MODEL_SERVICE" "$LLAMA_MODEL_PORT" "8080" "$LLAMA_MODEL_NAMESPACE" "Llama Model"
    else
        echo -e "${RED}❌  Llama Model service not found in namespace: $LLAMA_MODEL_NAMESPACE. Exiting...${NC}"
        exit 1
    fi
    
    # Find and port-forward TempoStack gateway
    TEMPO_SERVICE=$(oc get services -n "$OBSERVABILITY_NAMESPACE" -o name -l 'app.kubernetes.io/name=tempo' -l 'app.kubernetes.io/component=gateway' | cut -d'/' -f2 || echo "")
    if [ -n "$TEMPO_SERVICE" ]; then
        create_port_forward "service" "$TEMPO_SERVICE" "$TEMPO_PORT" "8080" "$OBSERVABILITY_NAMESPACE" "Tempo"
    else
        echo -e "${YELLOW}⚠️  TempoStack gateway not found in namespace: $OBSERVABILITY_NAMESPACE. Tempo functionality may not be available.${NC}"
    fi

    sleep 5  # Give port-forwards time to establish

    # Verify port-forwards are working
    echo -e "${BLUE}🔍 Verifying port-forwards...${NC}"
    verify_port_forward "Thanos" "https://localhost:$THANOS_PORT/api/v1/query?query=up" "$TOKEN"
    verify_port_forward "Tempo" "https://localhost:$TEMPO_PORT/api/traces/v1/dev/api/traces?service=test" "$TOKEN"
    verify_port_forward "LlamaStack" "http://localhost:$LLAMASTACK_PORT/v1/openai/v1/models" ""

    # Show port-forward status with resource details
    echo -e "${BLUE}🔍 Port-forward status:${NC}"
    echo -e "${YELLOW}   📊 Thanos: localhost:$THANOS_PORT → ${THANOS_SERVICE:-$THANOS_POD}${NC}"
    echo -e "${YELLOW}   🦙 LlamaStack: localhost:$LLAMASTACK_PORT → $LLAMASTACK_POD${NC}"
    echo -e "${YELLOW}   🤖 Llama Model: localhost:$LLAMA_MODEL_PORT → $LLAMA_MODEL_SERVICE${NC}"
    if [ -n "$TEMPO_SERVICE" ]; then
        echo -e "${YELLOW}   📊 Tempo: localhost:$TEMPO_PORT → $TEMPO_SERVICE${NC}"
    else
        echo -e "${YELLOW}   📊 Tempo: Not available${NC}"
    fi
}

# Function to verify port-forward is working
verify_port_forward() {
    local service_name="$1"
    local url="$2"
    local token="$3"

    local max_attempts=5
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if [ -n "$token" ]; then
            response=$(curl -k -s -w "%{http_code}" -H "Authorization: Bearer $token" "$url" -o /dev/null 2>/dev/null)
        else
            response=$(curl -k -s -w "%{http_code}" "$url" -o /dev/null 2>/dev/null)
        fi

        if [ "$response" = "200" ] || [ "$response" = "302" ] || [ "$response" = "401" ] || [ "$response" = "403" ]; then
            echo -e "${GREEN}   ✅ $service_name port-forward is working (HTTP $response)${NC}"
            return 0
        else
            echo -e "${YELLOW}   ⏳ $service_name port-forward attempt $attempt/$max_attempts (HTTP $response)${NC}"
            sleep 2
            attempt=$((attempt + 1))
        fi
    done

    echo -e "${RED}   ❌ $service_name port-forward failed after $max_attempts attempts${NC}"
    return 1
}

# Function to check if port-forwards are still active
check_port_forwards() {
    local all_active=true

    echo -e "${BLUE}🔍 Checking port-forward status...${NC}"

    # Check all port-forwards using helper function
    check_port_status "$THANOS_PORT" "Thanos" "${THANOS_SERVICE:-$THANOS_POD}" || all_active=false
    check_port_status "$TEMPO_PORT" "Tempo" "$TEMPO_SERVICE" || all_active=false
    check_port_status "$LLAMASTACK_PORT" "LlamaStack" "$LLAMASTACK_POD" || all_active=false
    check_port_status "$LLAMA_MODEL_PORT" "Llama Model" "$LLAMA_MODEL_SERVICE" || all_active=false

    if [ "$all_active" = false ]; then
        echo -e "${RED}❌ Some port-forwards are not active! This will cause connection failures.${NC}"
        echo -e "${YELLOW}💡 Try restarting the script or check for port conflicts.${NC}"
        return 1
    else
        echo -e "${GREEN}✅ All port-forwards are active${NC}"
        return 0
    fi
}

# Function to restart port-forwards if they fail
restart_port_forwards() {
    echo -e "${YELLOW}🔄 Restarting port-forwards...${NC}"

    # Kill existing port-forwards
    kill_port_forwards true
    sleep 2

    # Restart port-forwards
    start_port_forwards
}

# Ensure a TCP port is free by terminating any process listening on it
ensure_port_free() {
    local port=$1
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $port is in use. Attempting to free it...${NC}"
        # Try graceful termination first
        lsof -nP -iTCP:"$port" -sTCP:LISTEN -t | xargs -r kill || true
        sleep 1
        # Force kill if still listening
        if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            lsof -nP -iTCP:"$port" -sTCP:LISTEN -t | xargs -r kill -9 || true
        fi
        if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            echo -e "${RED}❌ Could not free port $port. Please free it and retry.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Port $port is now free${NC}"
    fi
}

# This function sets "MODEL_CONFIG" environment variable from cluster deployment or local file
set_model_config() {
    echo -e "${BLUE}🔧 Setting up MODEL_CONFIG...${NC}"
    echo -e "${BLUE}   Using config source: $MODEL_CONFIG_SOURCE${NC}"
    
    if [ "$MODEL_CONFIG_SOURCE" = "local" ]; then
        # Use local model config
        LOCAL_MODEL_CONFIG="deploy/helm/model-config.json"
        if [ -f "$LOCAL_MODEL_CONFIG" ]; then
            echo -e "${YELLOW}📋 Using LOCAL model config from: $LOCAL_MODEL_CONFIG${NC}"
            echo -e "${YELLOW}   This includes additional models like Anthropic Claude for testing.${NC}"
            export MODEL_CONFIG=$(cat "$LOCAL_MODEL_CONFIG")
            if [ -n "$MODEL_CONFIG" ]; then
                echo -e "${GREEN}✅ LOCAL MODEL_CONFIG loaded successfully${NC}"
                echo -e "${BLUE}   Available models: $(echo "$MODEL_CONFIG" | jq -r 'keys | join(", ")')${NC}"
                return 0
            else
                echo -e "${RED}❌ Failed to read local model config file${NC}"
                exit 1
            fi
        else
            echo -e "${RED}❌ Local model config file not found: $LOCAL_MODEL_CONFIG${NC}"
            echo -e "${YELLOW}   Please ensure the file exists or use cluster config with -c cluster${NC}"
            exit 1
        fi
    else
        # Use cluster config
        echo -e "${BLUE}🔧 Using CLUSTER model config...${NC}"
        METRIC_API_DEPLOYMENT=$(oc get deploy "$METRIC_API_APP" -n "$DEFAULT_NAMESPACE")
        if [ -n "$METRIC_API_DEPLOYMENT" ]; then
            echo -e "${YELLOW}✅ Found [$METRIC_API_APP] deployment:\n$METRIC_API_DEPLOYMENT${NC}"
            export $(oc set env deployment/$METRIC_API_APP --list  -n "$DEFAULT_NAMESPACE" | grep MODEL_CONFIG)
            if [ -n "$MODEL_CONFIG" ]; then
              echo -e "${GREEN}✅ CLUSTER MODEL_CONFIG set successfully${NC}"
            else
              echo -e "${RED}❌ Unable to set MODEL_CONFIG environment variable. It is required to run the UI locally.${NC}"
              exit 1
            fi
        else
            echo -e "${RED}❌ $METRIC_API_APP deployment not found. It is required to set MODEL_CONFIG.${NC}"
            exit 1
        fi
    fi
}

# Function to start local services
start_local_services() {
    echo -e "${BLUE}🏃 Starting local services...${NC}"
    
    # Get service account token
    TOKEN=$(oc whoami -t)
    
    # Set environment variables
    export PROMETHEUS_URL="https://localhost:$THANOS_PORT"
    export TEMPO_URL="https://localhost:$TEMPO_PORT"
    export TEMPO_TENANT_ID="dev"
    export TEMPO_TOKEN="$TOKEN"
    export LLAMA_STACK_URL="http://localhost:$LLAMASTACK_PORT/v1/openai/v1"
    export THANOS_TOKEN="$TOKEN"
    export METRICS_API_URL="http://localhost:$METRICS_API_PORT"
    export MCP_URL="http://localhost:$MCP_PORT"
    # PROM_URL is an alias for PROMETHEUS_URL (for backward compatibility)
    export PROM_URL="$PROMETHEUS_URL"
    export VERIFY_SSL=false
    export PYTHONHTTPSVERIFY=0
    export PYTHONWARNINGS="ignore:Unverified HTTPS request"
    # Set log level (override with PYTHON_LOG_LEVEL=DEBUG for more verbose logging)
    export PYTHON_LOG_LEVEL="${PYTHON_LOG_LEVEL:-INFO}"

    # macOS weasyprint support
    export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"

    set_model_config
    
    # Start Metrics API backend
    echo -e "${BLUE}🔧 Starting Metrics API backend...${NC}"
    ensure_port_free "$METRICS_API_PORT"
    (cd src/api && python -m uvicorn metrics_api:app --host 0.0.0.0 --port $METRICS_API_PORT --reload > log.txt) &
    MCP_PID=$!
    
    # Wait for Metrics API to start
    sleep 3
    
    # Test Metrics API service
    if curl -s --connect-timeout 5 "http://localhost:$METRICS_API_PORT/models" > /dev/null; then
        echo -e "${GREEN}✅ Metrics API backend started successfully${NC}"
    else
        echo -e "${RED}❌ Metrics API backend failed to start${NC}"
        exit 1
    fi

    # Start MCP server (configurable transport protocol)
    echo -e "${BLUE}🧩 Starting MCP Server ($MCP_TRANSPORT_PROTOCOL)...${NC}"
    ensure_port_free "$MCP_PORT"
    (cd src && \
      MCP_TRANSPORT_PROTOCOL="$MCP_TRANSPORT_PROTOCOL" \
      MODEL_CONFIG="$MODEL_CONFIG" \
      PROMETHEUS_URL="$PROMETHEUS_URL" \
      TEMPO_URL="$TEMPO_URL" \
      TEMPO_TENANT_ID="$TEMPO_TENANT_ID" \
      TEMPO_TOKEN="$TEMPO_TOKEN" \
      LLAMA_STACK_URL="$LLAMA_STACK_URL" \
      THANOS_TOKEN="$THANOS_TOKEN" \
      python3 -m mcp_server.main > mcp_log.txt) &
    MCP_SRV_PID=$!

    # Wait for MCP server to start
    sleep 3

    # Test MCP server health
    if curl -s --connect-timeout 5 "http://localhost:$MCP_PORT/health" | grep -q '"status"'; then
        echo -e "${GREEN}✅ MCP Server started successfully on port $MCP_PORT${NC}"
    else
        echo -e "${RED}❌ MCP Server failed to start${NC}"
        exit 1
    fi
    
    # Start Streamlit UI
    echo -e "${BLUE}🎨 Starting Streamlit UI...${NC}"
    (cd src/ui && \
      MCP_SERVER_URL="http://localhost:$MCP_PORT" \
      streamlit run ui.py --server.port $UI_PORT --server.address 0.0.0.0 --server.headless true) &
    UI_PID=$!
    
    # Wait for UI to start
    sleep 5
    
    echo -e "${GREEN}✅ All services started successfully!${NC}"
}

# Main execution
main() {
    parse_args "$@"
    check_prerequisites

    # Set cleanup trap only after successful prerequisite checks
    trap cleanup EXIT INT TERM

    echo ""
    echo -e "${BLUE}--------------------------------${NC}"
    echo -e "${BLUE}Configuration being used for setup:${NC}"
    echo -e "${BLUE}  DEFAULT_NAMESPACE: $DEFAULT_NAMESPACE${NC}"
    echo -e "${BLUE}  LLAMA_MODEL_NAMESPACE: $LLAMA_MODEL_NAMESPACE${NC}"
    echo -e "${BLUE}  MODEL_CONFIG_SOURCE: $MODEL_CONFIG_SOURCE${NC}"
    echo -e "${BLUE}--------------------------------${NC}\n"

    start_port_forwards
    start_local_services
    
    echo -e "\n${GREEN}🎉 Setup complete! All services are running.${NC}"
    echo -e "\n${BLUE}📋 Services Available:${NC}"
    echo -e "   ${YELLOW}🎨 Streamlit UI: http://localhost:$UI_PORT${NC}"
    echo -e "   ${YELLOW}🔧 Metrics API: http://localhost:$METRICS_API_PORT/docs${NC}"
    echo -e "   ${YELLOW}🧩 MCP Server (health): http://localhost:$MCP_PORT/health${NC}"
    echo -e "   ${YELLOW}🧩 MCP StreamableHTTP Endpoint: http://localhost:$MCP_PORT/mcp${NC}"
    echo -e "   ${YELLOW}📊 Prometheus: https://localhost:$THANOS_PORT${NC}"
    echo -e "   ${YELLOW}🔍 TempoStack: https://localhost:$TEMPO_PORT${NC}"
    echo -e "   ${YELLOW}🦙 LlamaStack: http://localhost:$LLAMASTACK_PORT${NC}"
    echo -e "   ${YELLOW}🤖 Llama Model: http://localhost:$LLAMA_MODEL_PORT${NC}"
    
    echo -e "\n${GREEN}🎯 Ready to use! Open your browser to http://localhost:$UI_PORT${NC}"
    echo -e "\n${YELLOW}📝 Note: Keep this terminal open to maintain all services${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all services and cleanup${NC}"
    
    # Start background port-forward monitoring
    echo -e "\n${BLUE}🔍 Starting port-forward monitoring...${NC}"
    (
        last_restart=0
        while true; do
            sleep 30  # Check every 30 seconds
            if ! check_port_forwards >/dev/null 2>&1; then
                current_time=$(date +%s)
                # Only restart if it's been more than 60 seconds since last restart
                if [ $((current_time - last_restart)) -gt 60 ]; then
                    echo -e "\n${RED}⚠️  Port-forward monitoring detected inactive port-forwards!${NC}"
                    echo -e "${YELLOW}🔄 Attempting to restart port-forwards...${NC}"
                    restart_port_forwards
                    last_restart=$current_time
                else
                    echo -e "\n${YELLOW}⚠️  Port-forwards inactive, but restart cooldown active${NC}"
                fi
            # Port-forwards are working - no need to log this
            fi
        done
    ) &
    MONITOR_PID=$!

    # Keep script running
    wait
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run main function
main "$@"