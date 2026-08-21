"""VoiceRAG — Streamlit app for HuggingFace Spaces"""

import os
import sys
from pathlib import Path

# Set paths
_PKG_ROOT = Path(__file__).resolve().parent / "voicerag"
os.environ["HF_HOME"] = str(_PKG_ROOT / "data" / "hf_cache")
os.environ["VECTOR_DB_PATH"] = str(_PKG_ROOT / "data" / "index")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import time

st.title("🎙️ VoiceRAG")
st.caption("Hindi/English question answering with RAG")

query = st.text_input("Ask a question", placeholder="गोवा कहाँ है?")

if query:
    with st.spinner("Thinking..."):
        try:
            from voicerag.harness.retriever import HybridRetriever
            from voicerag.harness.stages import detect_language, normalize_query, guardrail_check, generate

            t0 = time.perf_counter()
            lang = detect_language(query)
            lang_name = {"hi": "Hindi", "en": "English", "ur": "Urdu"}.get(lang, lang)

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
            t_retrieve = (time.perf_counter() - t0) * 1000

            gr_result, _ = guardrail_check(query, res)
            if gr_result.refused:
                st.warning("No information found.")
            else:
                gen, _ = generate(query, res, gr_result, timeout_ms=15000)
                t_total = (time.perf_counter() - t0) * 1000

                if lang == "en" and gen.answer:
                    try:
                        from voicerag.harness.stages import translate
                        gen.answer, _ = translate(gen.answer, "hi", "en")
                    except:
                        pass

                st.success(gen.answer)

                col1, col2, col3 = st.columns(3)
                col1.metric("Language", lang_name)
                col2.metric("Confidence", f"{gen.confidence:.0%}")
                col3.metric("Latency", f"{t_total:.0f}ms")

                st.caption(f"Citations: {', '.join(gen.citations)}")

        except Exception as e:
            st.error(f"Error: {e}")
