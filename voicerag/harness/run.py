"""
Alias entry point so `python -m harness.run --query "..."` works as
documented in AGENTS.md §6 (same as `python -m harness`).
"""

from .__main__ import main

if __name__ == "__main__":
    main()
