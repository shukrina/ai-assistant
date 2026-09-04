"""
Streamlit chat UI for the AI Assistant backend.

Run:  streamlit run ui/streamlit_app.py
Configure the backend URL with the API_BASE_URL environment variable
(defaults to http://localhost:8000, or http://app:8000 inside Docker Compose).
"""
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")
st.caption("RAG-powered chat assistant with tool calling, caching, and provider fallback.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Settings")
    use_rag = st.toggle("Use knowledge base (RAG)", value=True)
    allow_tools = st.toggle("Allow tool calling", value=True)

    st.divider()
    st.subheader("📄 Ingest a document")
    uploaded = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
    if uploaded and st.button("Ingest"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        try:
            resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Ingested {data['filename']} → {data['num_chunks']} chunks")
        except requests.RequestException as exc:
            st.error(f"Ingestion failed: {exc}")

    st.divider()
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.caption(f"Backend status: **{health['status']}**")
    except requests.RequestException:
        st.caption("Backend status: ⚠️ unreachable")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("📚 Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**{s['document']}** (score {s['score']})\n\n> {s['text'][:300]}...")
        if turn.get("tool_calls"):
            with st.expander(f"🛠️ Tool calls ({len(turn['tool_calls'])})"):
                for tc in turn["tool_calls"]:
                    st.markdown(f"**{tc['name']}**({tc['arguments']}) → `{tc['result']}`")
        if turn.get("provider_used"):
            badge = "🟡 fallback" if turn.get("used_fallback") else "🟢 primary"
            st.caption(f"{badge} · {turn['provider_used']} · {turn.get('latency_ms', '?')} ms")

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "use_rag": use_rag,
                        "allow_tools": allow_tools,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.session_id = data["session_id"]

                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("📚 Sources"):
                        for s in data["sources"]:
                            st.markdown(f"**{s['document']}** (score {s['score']})\n\n> {s['text'][:300]}...")
                if data.get("tool_calls"):
                    with st.expander(f"🛠️ Tool calls ({len(data['tool_calls'])})"):
                        for tc in data["tool_calls"]:
                            st.markdown(f"**{tc['name']}**({tc['arguments']}) → `{tc['result']}`")

                badge = "🟡 fallback" if data.get("used_fallback") else "🟢 primary"
                cache_note = " · from cache" if data.get("cached") else ""
                st.caption(f"{badge} · {data['provider_used']} · {data['latency_ms']} ms{cache_note}")

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "tool_calls": data.get("tool_calls", []),
                        "provider_used": data.get("provider_used"),
                        "used_fallback": data.get("used_fallback"),
                        "latency_ms": data.get("latency_ms"),
                    }
                )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
