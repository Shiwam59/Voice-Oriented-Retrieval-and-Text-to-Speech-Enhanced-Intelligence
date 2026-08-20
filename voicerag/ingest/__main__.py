"""
Ingest module — entry point for building the index end-to-end.

Usage:
    python -m ingest.build_index --split hi
"""

from .build_index import main

if __name__ == "__main__":
    main()
