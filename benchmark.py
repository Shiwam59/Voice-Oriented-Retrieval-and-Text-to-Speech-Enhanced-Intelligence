"""Convenience entry point for the full VoiceRAG harness benchmark.

Run from the repository root::

    python benchmark.py --n 20 --warmup 2

The implementation lives in ``voicerag.eval.benchmark``.
"""

from voicerag.eval.benchmark import main


if __name__ == "__main__":
    main()
