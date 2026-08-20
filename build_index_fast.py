"""
Fast index build script — use passage-native chunks only.
This takes ~5 minutes instead of 2 hours.
"""

import logging
import os
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def build_fast_index():
    """Build index from passage-native chunks only."""
    raw_dir = Path("./data/raw")
    index_dir = Path("./data/index")
    
    logger.info("=" * 60)
    logger.info("FAST INDEX BUILD (passage-native chunks only)")
    logger.info("=" * 60)
    
    # Step 1: Merge existing chunks into consolidated chunks.parquet
    logger.info("Step 1: Preparing chunks...")
    chunks_passage = index_dir / "chunks_passage_native.parquet"
    
    if chunks_passage.exists():
        logger.info(f"  Loading {chunks_passage}...")
        chunks_df = pd.read_parquet(chunks_passage)
        logger.info(f"  ✓ Loaded {len(chunks_df)} passage-native chunks")
        
        # Save as consolidated chunks.parquet for build_all
        chunks_consolidated = index_dir / "chunks.parquet"
        chunks_df.to_parquet(chunks_consolidated)
        logger.info(f"  ✓ Saved consolidated chunks to {chunks_consolidated}")
    else:
        raise FileNotFoundError(f"Chunks not found at {chunks_passage}")
    
    # Step 2: Build indices
    logger.info("=" * 60)
    logger.info("Step 2: Building FAISS + BM25 + metadata...")
    from voicerag.ingest.build_index import build_all
    
    try:
        build_all(raw_dir, index_dir)
        logger.info("✓ Index built successfully!")
        logger.info(f"  Index dir: {index_dir}")
        logger.info("  FAISS index: faiss_hnsw.index")
        logger.info("  BM25 index: bm25_retriever.pkl")
        logger.info("  Metadata: metadata.db")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        raise

if __name__ == "__main__":
    build_fast_index()
