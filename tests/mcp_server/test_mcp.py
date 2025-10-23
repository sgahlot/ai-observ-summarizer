"""Simple tests for ObservabilityMCP server functionality."""

import sys
import os

# Python path is now handled by conftest.py


def test_observability_mcp_server_imports():
    """Test that ObservabilityMCPServer can be imported."""
    try:
        from mcp_server.observability_mcp import ObservabilityMCPServer
        assert ObservabilityMCPServer is not None
    except ImportError as e:
        assert False, f"Failed to import ObservabilityMCPServer: {e}"


def test_observability_mcp_server_class_exists():
    """Test that ObservabilityMCPServer class has expected methods."""
    from mcp_server.observability_mcp import ObservabilityMCPServer

    # Check that the class has the expected methods
    assert hasattr(ObservabilityMCPServer, '__init__')
    assert hasattr(ObservabilityMCPServer, '_register_mcp_tools')


