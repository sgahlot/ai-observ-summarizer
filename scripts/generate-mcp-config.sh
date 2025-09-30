#!/bin/bash

# AI Observability MCP Configuration Generator
# This script generates fresh MCP configuration files for Cursor and Claude Desktop with current OpenShift token

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$SCRIPT_DIR/mcp-config-templates"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c TARGET                Generate configuration for specified target"
    echo "                           Valid targets: 'cursor', 'claude', 'all' (case-insensitive, default: all)"
    echo "  -h                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Generate both Cursor and Claude configs (default)"
    echo "  $0 -c cursor             # Generate Cursor config only"
    echo "  $0 -c CURSOR             # Generate Cursor config only (uppercase)"
    echo "  $0 -c claude             # Generate Claude Desktop config only"
    echo "  $0 -c CLAUDE             # Generate Claude Desktop config only (uppercase)"
    echo "  $0 -c all                # Generate both Cursor and Claude configs"
}

# Function to parse command line arguments
parse_args() {
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        usage
        exit 0
    fi

    TARGET="all"

    while getopts "c:C:hH" opt; do
        case $opt in
            c|C) TARGET="$OPTARG"
               # Normalize target to lowercase for case-insensitive comparison
               TARGET_LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')

               # Validate target (case-insensitive)
               if [[ "$TARGET_LOWER" != "cursor" && "$TARGET_LOWER" != "claude" && "$TARGET_LOWER" != "all" ]]; then
                  echo -e "${RED}❌ Invalid target: $TARGET${NC}"
                  echo -e "${YELLOW}   Valid targets: cursor, claude, all (case-insensitive)${NC}"
                  usage
                  exit 1
               fi

               # Use normalized target for consistency
               TARGET="$TARGET_LOWER"
               ;;
            h|H) usage
               exit 0
               ;;
            *) echo -e "${RED}❌ INVALID option: [$OPTARG]${NC}"
               usage
               exit 1
               ;;
        esac
    done
}

echo -e "${BLUE}🔧 AI Observability MCP Configuration Generator${NC}"
echo "=================================================="

# Parse command line arguments
parse_args "$@"

# Detect operating system and set config paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CURSOR_CONFIG_DIR="$HOME/.cursor"
    CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CURSOR_CONFIG_DIR="$HOME/.cursor"
    CLAUDE_CONFIG_DIR="$HOME/.config/claude"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    CURSOR_CONFIG_DIR="$APPDATA/Cursor"
    CLAUDE_CONFIG_DIR="$APPDATA/Claude"
else
    echo -e "${RED}❌ Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

CURSOR_CONFIG="$CURSOR_CONFIG_DIR/mcp.json"
CLAUDE_CONFIG="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

# Check if template files exist
if [ ! -d "$TEMPLATES_DIR" ]; then
    echo -e "${RED}❌ Template directory not found: $TEMPLATES_DIR${NC}"
    exit 1
fi

# Check if OpenShift CLI is available
if ! command -v oc &> /dev/null; then
    echo -e "${RED}❌ OpenShift CLI (oc) is not installed${NC}"
    echo -e "${YELLOW}   Please install OpenShift CLI first${NC}"
    exit 1
fi


# Get project base directory (for reference)
OBS_BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo -e "${BLUE}📁 Project directory: $OBS_BASE_DIR${NC}"

# Check if local MCP server binary exists (needed for both Cursor and Claude Desktop)
MCP_SERVER_BINARY="$OBS_BASE_DIR/.venv/bin/obs-mcp-server"
if [ ! -f "$MCP_SERVER_BINARY" ]; then
    echo -e "${RED}❌ MCP server binary not found: $MCP_SERVER_BINARY${NC}"
    echo -e "${YELLOW}   Please ensure the virtual environment is set up and the MCP server is installed${NC}"
    echo -e "${YELLOW}   Run: cd $OBS_BASE_DIR && uv sync && uv run pip install -e .${NC}"
    exit 1
fi
echo -e "${GREEN}✅ MCP server binary found: $MCP_SERVER_BINARY${NC}"

# Check if logged in to OpenShift
if ! oc whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged in to OpenShift cluster${NC}"
    echo -e "${YELLOW}   Please run: oc login${NC}"
    exit 1
fi

# Get fresh token
echo -e "${BLUE}🔑 Getting OpenShift token...${NC}"
export OC_TOKEN=$(oc whoami -t)

if [ -z "$OC_TOKEN" ]; then
    echo -e "${RED}❌ Failed to get OpenShift token${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Got OpenShift token${NC}"

# Function to generate config from template
generate_config() {
    local template_file="$1"
    local output_file="$2"
    local config_name="$3"
    
    if [ ! -f "$template_file" ]; then
        echo -e "${RED}❌ Template file not found: $template_file${NC}"
        return 1
    fi
    
    echo -e "${BLUE}⚙️  Generating $config_name configuration...${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"
    
    # Backup existing config if it exists
    if [ -f "$output_file" ]; then
        echo -e "${YELLOW}📋 Backing up existing $config_name config...${NC}"
        cp "$output_file" "$output_file.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Generate config from template with variable substitution using envsubst
    # Export variables for envsubst
    export OC_TOKEN
    export OBS_BASE_DIR

    # Use envsubst to substitute environment variables
    envsubst < "$template_file" > "$output_file"

    # Validate JSON
    if command -v python3 &> /dev/null; then
        echo -e "${BLUE}🔍 Validating $config_name JSON configuration...${NC}"
        if python3 -m json.tool "$output_file" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $config_name configuration is valid JSON${NC}"
        else
            echo -e "${RED}❌ $config_name configuration is invalid JSON${NC}"
            return 1
        fi
    fi
    
    echo -e "${GREEN}✅ $config_name configuration generated: $output_file${NC}"
    return 0
}

# Generate configurations based on target
SUCCESS_COUNT=0
TOTAL_COUNT=0

if [[ "$TARGET" == "cursor" || "$TARGET" == "all" ]]; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if generate_config "$TEMPLATES_DIR/cursor-config.json.template" "$CURSOR_CONFIG" "Cursor"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "\n${BLUE}📋 Cursor Configuration:${NC}"
        echo -e "   📁 Config file: $CURSOR_CONFIG"
        echo -e "   🔧 Command: $OBS_BASE_DIR/.venv/bin/obs-mcp-server --local"
    fi
fi

if [[ "$TARGET" == "claude" || "$TARGET" == "all" ]]; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if generate_config "$TEMPLATES_DIR/claude-config.json.template" "$CLAUDE_CONFIG" "Claude Desktop"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "\n${BLUE}📋 Claude Desktop Configuration:${NC}"
        echo -e "   📁 Config file: $CLAUDE_CONFIG"
        echo -e "   🔧 Command: $OBS_BASE_DIR/.venv/bin/obs-mcp-server --local"
    fi
fi

# Summary
echo -e "\n${GREEN}🎉 MCP Configuration Generation Complete!${NC}"
echo -e "\n${BLUE}📋 Summary:${NC}"
echo -e "   ✅ Successfully generated: $SUCCESS_COUNT/$TOTAL_COUNT configurations"
echo -e "   🔑 Token: ${OC_TOKEN:0:20}... (truncated for security)"
echo -e "   ⏰ Generated: $(date)"

echo -e "\n${YELLOW}📝 Next Steps:${NC}"
echo -e "   1. 🔄 Restart your application(s) to load the new configuration(s)"
echo -e "   2. 🚀 Start local development environment:"
echo -e "      cd $OBS_BASE_DIR"
echo -e "      scripts/local-dev.sh -n <your-namespace>"
echo -e "   3. 🧪 Test MCP tools with queries like:"
echo -e "      - \"What models are available?\""
echo -e "      - \"Show me traces for service llama-3-2-3b-instruct-predictor\""
echo -e "      - \"Analyze model performance for the last hour\""

echo -e "\n${BLUE}💡 Pro Tips:${NC}"
echo -e "   Create aliases for convenience:"
echo -e "   alias start-cursor-mcp=\"$SCRIPT_DIR/generate-mcp-config.sh -c cursor && open -a Cursor\""
echo -e "   alias start-claude-mcp=\"$SCRIPT_DIR/generate-mcp-config.sh -c claude && open -a Claude\""
echo -e "   alias start-all-mcp=\"$SCRIPT_DIR/generate-mcp-config.sh -c all\""

echo -e "\n${YELLOW}⚠️  Notes:${NC}"
echo -e "   - Token expires in ~24 hours, re-run this script when needed"
echo -e "   - Template files can be edited in: $TEMPLATES_DIR"
echo -e "   - The local-dev.sh script must be running for MCP tools to work"
echo -e "   - Use --help to see all available options"
