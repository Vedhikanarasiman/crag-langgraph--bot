import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.retrieval.retriever import retrieve_chunks
from src.graph.state import GraphState
from duckduckgo_search import DDGS

load_dotenv()

# Initialize Gemini
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ─────────────────────────────────────────
# NODE 1: Query Rewriter
# ─────────────────────────────────────────
def query_rewriter(state: GraphState) -> GraphState:
    """
    Rewrites the user's question into a better
    search query for semantic search.
    """
    print("--- Node 1: Query Rewriter ---")

    question = state["question"]

    prompt = f"""You are an expert at rewriting questions for semantic search 
over AI research papers.

Rewrite this question into a detailed search query that will retrieve 
the most relevant content from research papers.
Return ONLY the rewritten query, nothing else.

Question: {question}
Rewritten query:"""

    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    print(f"Original: {question}")
    print(f"Rewritten: {rewritten}")

    return {**state, "rewritten_query": rewritten}


# ─────────────────────────────────────────
# NODE 2: Retriever
# ─────────────────────────────────────────
def retriever_node(state: GraphState) -> GraphState:
    """
    Uses the rewritten query to search ChromaDB
    and retrieve top 5 relevant chunks.
    """
    print("--- Node 2: Retriever ---")

    query = state["rewritten_query"]
    chunks = retrieve_chunks(query, k=5)

    print(f"Retrieved {len(chunks)} chunks")

    return {**state, "retrieved_chunks": chunks}


# ─────────────────────────────────────────
# NODE 3: Relevance Grader
# ─────────────────────────────────────────
def relevance_grader(state: GraphState) -> GraphState:
    """
    Checks each retrieved chunk — is it actually
    relevant to the user's question?
    Filters out irrelevant chunks.
    """
    print("--- Node 3: Relevance Grader ---")

    question = state["question"]
    chunks = state["retrieved_chunks"]
    filtered = []

    for chunk in chunks:
        prompt = f"""You are grading whether a document chunk is relevant 
to a question.

Question: {question}
Chunk: {chunk['text']}

Is this chunk relevant to answering the question?
Reply with ONLY 'relevant' or 'not_relevant'."""

        response = llm.invoke(prompt)
        score = response.content.strip().lower()

        if "relevant" in score and "not" not in score:
            filtered.append(chunk)
            print(f"Chunk from {chunk['metadata']['source']} p{chunk['metadata']['page']}: relevant")
        else:
            print(f"Chunk from {chunk['metadata']['source']} p{chunk['metadata']['page']}: not relevant")

    # Decide overall relevance
    relevance_score = "relevant" if len(filtered) > 0 else "not_relevant"
    print(f"Relevance decision: {relevance_score}")

    return {**state, "filtered_chunks": filtered, "relevance_score": relevance_score}


# ─────────────────────────────────────────
# NODE 4: Web Fallback
# ─────────────────────────────────────────
def web_fallback(state: GraphState) -> GraphState:
    """
    If no relevant chunks found in ChromaDB,
    search the web using DuckDuckGo.
    """
    print("--- Node 4: Web Fallback ---")

    query = state["rewritten_query"]

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    # Convert web results to same format as chunks
    web_chunks = []
    for r in results:
        web_chunks.append({
            "text": r.get("body", ""),
            "metadata": {
                "source": r.get("href", "web"),
                "page": "web"
            }
        })

    print(f"Found {len(web_chunks)} web results")

    return {**state, "filtered_chunks": web_chunks}


# ─────────────────────────────────────────
# NODE 5: Answer Generator
# ─────────────────────────────────────────
def answer_generator(state: GraphState) -> GraphState:
    """
    Uses Gemini to generate a final answer
    strictly based on the filtered chunks.
    Includes source citations.
    """
    print("--- Node 5: Answer Generator ---")

    question = state["question"]
    chunks = state["filtered_chunks"]
    chat_history = state.get("chat_history", [])

    # Format chunks as context
    context = ""
    sources = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        context += f"\n[Source: {source}, Page: {page}]\n{chunk['text']}\n"
        source_ref = f"{source} (Page {page})"
        if source_ref not in sources:
            sources.append(source_ref)

    # Format chat history
    history_text = ""
    for msg in chat_history[-4:]:  # last 2 exchanges
        if hasattr(msg, "type"):
            history_text += f"{msg.type}: {msg.content}\n"

    prompt = f"""You are an expert AI research assistant with deep knowledge 
of machine learning and NLP papers.

Use the provided context as your PRIMARY source. Where the context gives 
specific details, cite them. Where the context is limited, you can 
supplement with your general knowledge about the topic but clearly 
indicate when you are doing so.

Be detailed, clear, and helpful. Structure your answer with explanation 
first, then specifics.

Previous conversation:
{history_text}

Context from research papers:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    print(f"Answer generated: {answer[:100]}...")

    return {**state, "answer": answer, "sources": sources}


# ─────────────────────────────────────────
# NODE 6: Hallucination Checker
# ─────────────────────────────────────────
def hallucination_checker(state: GraphState) -> GraphState:
    print("--- Node 6: Hallucination Checker ---")

    answer = state["answer"]
    chunks = state["filtered_chunks"]
    retry_count = state.get("retry_count", 0)

    # If we've retried 2 times already, just accept the answer
    if retry_count >= 2:
        print("Max retries reached, accepting answer")
        return {**state, "hallucination_flag": "grounded", "retry_count": 0}

    context = "\n".join([chunk["text"] for chunk in chunks])

    prompt = f"""You are checking if an answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context?
Reply with ONLY 'grounded' or 'not_grounded'."""

    response = llm.invoke(prompt)
    flag = response.content.strip().lower()

    if "not_grounded" in flag or "not grounded" in flag:
        hallucination_flag = "not_grounded"
    else:
        hallucination_flag = "grounded"

    print(f"Hallucination check: {hallucination_flag} (retry {retry_count})")

    return {
        **state,
        "hallucination_flag": hallucination_flag,
        "retry_count": retry_count + 1
    }