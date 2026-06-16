import streamlit as st
from src.graph.graph import build_graph

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────
# INITIALIZE SESSION STATE
# ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session-1"

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("ResearchMind")
    st.markdown("Chat with 15 AI Research Papers")
    st.divider()

    st.subheader("📚 Loaded Papers")
    papers = [
        "Attention is All You Need",
        "SELF-RAG",
        "HyDE",
        "ReAct",
        "Reflexion",
        "GraphRAG",
        "Chain of Thought",
        "RAG vs Fine-tuning",
        "LLM as Judge",
        "Corrective RAG",
        "Dense Passage Retrieval",
        "RAPTOR",
        "Mistral 7B",
        "LLM Agents Survey",
        "LangGraph Agent"
    ]
    for paper in papers:
        st.markdown(f"📄 {paper}")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.thread_id = "session-1"
        st.rerun()

# ─────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────
st.title("ResearchMind")
st.caption("Ask anything about AI research papers")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
if question := st.chat_input("Ask a question about the research papers..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Add to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }

                result = st.session_state.graph.invoke(
                    {
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                        "retry_count": 0
                    },
                    config=config
                )

                answer = result["answer"]
                sources = result.get("sources", [])

                # Display answer
                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")