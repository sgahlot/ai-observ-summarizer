# MCP Server Setup Guide

This guide explains how to set up the MCP (Model Context Protocol) server for local development.

## Prerequisites

- Python 3.10+
- `uv` package manager installed
- OpenShift CLI (`oc`) for token generation

## Quick Setup (From Scratch)

### 1. Navigate to Project Directory
```bash
cd /path/to/summarizer
```

### 2. Create Virtual Environment and Install Dependencies
```bash
uv sync
```
This creates the `.venv` directory and installs all dependencies from `pyproject.toml` and `uv.lock`.

### 3. Install MCP Server Package
```bash
uv add --editable src/mcp_server
```
This installs the MCP server package in editable mode and creates console scripts.

### 4. Verify Installation
```bash
source .venv/bin/activate
which obs-mcp-server
# Should show: /path/to/summarizer/.venv/bin/obs-mcp-server

which obs-mcp-stdio
# Should show: /path/to/summarizer/.venv/bin/obs-mcp-stdio
```

### 5. Test the MCP Server
```bash
obs-mcp-server --help
```

## Available Commands

- **`obs-mcp-server`**: Main MCP server executable
- **`obs-mcp-stdio`**: STDIO-based MCP server

## Configuration

The MCP server can be configured using:
- Environment variables (see `src/mcp_server/env.template`)
- Command-line arguments
- Configuration files

## Troubleshooting

### Issue: Executables not found
**Problem**: `which obs-mcp-server` returns nothing
**Solution**: Ensure you're using `uv add --editable` instead of `pip install -e`

### Issue: Wrong installation location
**Problem**: Executables installed in system Python instead of virtual environment
**Solution**: Use `uv` commands instead of `pip` when `uv` is managing the project

### Issue: ModuleNotFoundError for 'common' package
**Problem**: `ModuleNotFoundError: No module named 'common'` or `No module named 'mcp_server.common'`
**Solution**: The `common` package needs to be included in the MCP server setup. This is already fixed in the current setup.py configuration.

### Issue: Virtual environment not activated
**Problem**: Commands not working after setup
**Solution**: Either activate the venv with `source .venv/bin/activate` or use `uv run <command>`

## Development Workflow

1. Make changes to MCP server code in `src/mcp_server/`
2. Changes are automatically reflected (editable install)
3. Test with `obs-mcp-server --help` or run the server
4. Use `uv run` to execute commands in the virtual environment

## Integration with AI Clients

The MCP server can be integrated with:
- **Cursor**: Via MCP configuration
- **Claude Desktop**: Via MCP configuration
- **Other MCP clients**: Using the standard MCP protocol

See `scripts/generate-mcp-config.sh` for generating client configurations.
