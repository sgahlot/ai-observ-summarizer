"""Pytest configuration for the observability summarizer tests."""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Also add the project root to the path for absolute imports
sys.path.insert(0, str(project_root))
