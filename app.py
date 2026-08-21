"""
VoiceRAG — Simple Gradio app for HuggingFace Spaces
"""

import os
import sys
from pathlib import Path

# Set paths before any imports
_PKG_ROOT = Path(__file__).resolve().parent / "voicerag"
os.environ["HF_HOME"] = str(_PKG_ROOT / "data" / "hf_cache")
os.environ["VECTOR_DB_PATH"] = str(_PKG_ROOT / "data" / "index")

import gradio as gr
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def answer(query: str) -> str:
    """Run the RAG pipeline and return formatted answer."""
    if not query.strip():
        return "Please type a question."

    try:
        from voicerag.harness.retriever import HybridRetriever
        from voicerag.harness.stages import (
            detect_language,
            normalize_query,
            guardrail_check,
            generate,
        )

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
                search_query = query

        # Normalize
        q_norm, _ = normalize_query(search_query)

        # Retrieve
        retriever = HybridRetriever(os.environ["VECTOR_DB_PATH"])
        res = retriever.retrieve(q_norm, top_k=5)
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
                pass

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
)

if __name__ == "__main__":
    demo.launch()
