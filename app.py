"""VoiceRAG for HuggingFace Spaces"""

import os
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent / "voicerag"
os.environ["HF_HOME"] = str(_PKG_ROOT / "data" / "hf_cache")
os.environ["VECTOR_DB_PATH"] = str(_PKG_ROOT / "data" / "index")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

def answer(query):
    if not query:
        return "Please type a question."
    try:
        from voicerag.harness.retriever import HybridRetriever
        from voicerag.harness.stages import detect_language, normalize_query, guardrail_check, generate

        lang = detect_language(query)
        search_query = query
        if lang == "en":
            try:
                from voicerag.harness.stages import translate
                search_query, _ = translate(query, "en", "hi")
            except:
                pass

        q_norm, _ = normalize_query(search_query)
        retriever = HybridRetriever(os.environ["VECTOR_DB_PATH"])
        res = retriever.retrieve(q_norm, top_k=5)

        gr_result, _ = guardrail_check(query, res)
        if gr_result.refused:
            return "No information found."

        gen, _ = generate(query, res, gr_result, timeout_ms=15000)

        if lang == "en" and gen.answer:
            try:
                from voicerag.harness.stages import translate
                gen.answer, _ = translate(gen.answer, "hi", "en")
            except:
                pass

        return f"**{gen.answer}**\n\nCitations: {', '.join(gen.citations)}"
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(fn=answer, inputs="text", outputs="text", title="VoiceRAG")
demo.launch()
