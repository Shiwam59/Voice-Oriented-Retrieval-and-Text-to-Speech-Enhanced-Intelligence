"""
VoiceRAG — Gradio Blocks app for HuggingFace Spaces
"""

import os
import sys
from pathlib import Path

# Set paths
_PKG_ROOT = Path(__file__).resolve().parent / "voicerag"
os.environ["HF_HOME"] = str(_PKG_ROOT / "data" / "hf_cache")
os.environ["VECTOR_DB_PATH"] = str(_PKG_ROOT / "data" / "index")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import time


def answer(query: str) -> str:
    """Run the RAG pipeline."""
    if not query or not query.strip():
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
        lang = detect_language(query)
        lang_name = {"hi": "Hindi", "en": "English", "ur": "Urdu"}.get(lang, lang)

        search_query = query
        if lang == "en":
            try:
                from voicerag.harness.stages import translate
                search_query, _ = translate(query, "en", "hi")
            except Exception:
                pass

        q_norm, _ = normalize_query(search_query)
        retriever = HybridRetriever(os.environ["VECTOR_DB_PATH"])
        res = retriever.retrieve(q_norm, top_k=5)
        t_retrieve = (time.perf_counter() - t0) * 1000

        gr_result, _ = guardrail_check(query, res)
        if gr_result.refused:
            return f"❓ No information found. Language: {lang_name}"

        gen, _ = generate(query, res, gr_result, timeout_ms=15000)
        t_total = (time.perf_counter() - t0) * 1000

        if lang == "en" and gen.answer:
            try:
                from voicerag.harness.stages import translate
                gen.answer, _ = translate(gen.answer, "hi", "en")
            except Exception:
                pass

        citations = ", ".join(gen.citations) if gen.citations else "N/A"
        method = "Extractive" if gen.extractive else "LLM"

        return f"""**Answer:** {gen.answer}

**Language:** {lang_name} | **Confidence:** {gen.confidence:.0%} | **Method:** {method}
**Citations:** {citations} | **Latency:** {t_total:.0f}ms"""

    except Exception as e:
        return f"Error: {str(e)}"


# Build with Blocks
with gr.Blocks(title="VoiceRAG") as demo:
    gr.Markdown("# 🎙️ VoiceRAG\nHindi/English question answering with RAG")
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Your question", placeholder="गोवा कहाँ है?")
            btn = gr.Button("Ask")
        with gr.Column():
            output = gr.Markdown(label="Answer")

    btn.click(fn=answer, inputs=input_box, outputs=output)
    input_box.submit(fn=answer, inputs=input_box, outputs=output)

demo.launch()
