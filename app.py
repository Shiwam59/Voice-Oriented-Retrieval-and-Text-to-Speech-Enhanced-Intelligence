"""
VoiceRAG — Gradio app for HuggingFace Spaces
Bilingual (Hindi/English) voice-powered question answering.
"""

import os
import sys
from pathlib import Path

# Set paths before any imports
_PKG_ROOT = Path(__file__).resolve().parent / "voicerag"
os.environ.setdefault("HF_HOME", str(_PKG_ROOT / "data" / "hf_cache"))
os.environ["VECTOR_DB_PATH"] = str(_PKG_ROOT / "data" / "index")

import gradio as gr
import time

# Lazy-load the pipeline (heavy imports)
_retriever = None
_pipeline_ready = False


def _init_pipeline():
    """Initialize the RAG pipeline (downloads model on first run)."""
    global _retriever, _pipeline_ready
    if _pipeline_ready:
        return True
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))

        from voicerag.harness.retriever import HybridRetriever
        from voicerag.ingest.build_index import build_index

        index_dir = os.environ["VECTOR_DB_PATH"]
        index_path = Path(index_dir) / "faiss_hnsw.index"

        if not index_path.exists():
            print("[setup] Building index for first time...")
            Path(index_dir).mkdir(parents=True, exist_ok=True)
            raw_dir = _PKG_ROOT / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = raw_dir / "msmarco_xi_hi.parquet"
            if not parquet_path.exists():
                print("[setup] Downloading dataset...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="ai4bharat/MSMARCO-XI",
                    filename="train/hintrain.parquet",
                    repo_type="dataset",
                    local_dir=str(raw_dir),
                    local_dir_use_symlinks=False,
                )
            build_index()
            print("[setup] Index built!")

        _retriever = HybridRetriever(index_dir)
        _pipeline_ready = True
        return True
    except Exception as e:
        print(f"[error] Pipeline init failed: {e}")
        return False


def answer(query: str) -> str:
    """Run the RAG pipeline and return formatted answer."""
    if not query.strip():
        return "Please type a question."

    # Init pipeline on first query
    if not _pipeline_ready:
        status = _init_pipeline()
        if not status:
            return "⚠️ Pipeline is still loading. Try again in 30 seconds."

    try:
        from voicerag.harness.stages import (
            detect_language,
            normalize_query,
            guardrail_check,
            generate,
        )
        from voicerag.harness.retriever import HybridRetriever

        t0 = time.perf_counter()

        # Detect language
        lang = detect_language(query)
        lang_name = {"hi": "Hindi", "en": "English", "ur": "Urdu"}.get(lang, lang)

        # Translate English → Hindi for search
        search_query = query
        if lang == "en":
            try:
                from voicerag.harness.stages import translate
                search_query, _ = translate(query, "en", "hi")
            except Exception:
                search_query = query  # Fallback: search with English

        # Normalize
        q_norm, _ = normalize_query(search_query)

        # Retrieve
        res = _retriever.retrieve(q_norm, top_k=5)
        t_retrieve = (time.perf_counter() - t0) * 1000

        # Guardrail
        gr_result, _ = guardrail_check(query, res)

        if gr_result.refused:
            return f"❓ Sorry, I don't have information about this topic.\n\nDetected language: {lang_name}"

        # Generate answer
        gen, _ = generate(query, res, gr_result, timeout_ms=15000)
        t_total = (time.perf_counter() - t0) * 1000

        # Translate answer back to English if needed
        if lang == "en" and gen.answer:
            try:
                from voicerag.harness.stages import translate
                gen.answer, _ = translate(gen.answer, "hi", "en")
            except Exception:
                pass  # Return Hindi answer

        # Format output
        citations = ", ".join(gen.citations) if gen.citations else "N/A"
        path_type = "📝 Extractive" if gen.extractive else "🤖 LLM"

        output = f"""## Answer
{gen.answer}

---
| Detail | Value |
|--------|-------|
| **Language** | {lang_name} |
| **Confidence** | {gen.confidence:.0%} |
| **Method** | {path_type} |
| **Citations** | {citations} |
| **Latency** | {t_total:.0f}ms (retrieve: {t_retrieve:.0f}ms) |
"""
        return output

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# Build Gradio interface
demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        label="Ask a question (Hindi or English)",
        placeholder="गोवा कहाँ है? / Where is Goa?",
        lines=2,
    ),
    outputs=gr.Markdown(label="Answer"),
    title="🎙️ VoiceRAG",
    description="Bilingual (Hindi/English) question answering powered by hybrid retrieval (FAISS + BM25). Ask anything about India!",
    examples=[
        "गोवा कहाँ है?",
        "Where is ISRO?",
        "महात्मा गांधी कौन थे?",
        "What is Chandrayaan-3?",
        "भारत की राजधानी क्या है?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
