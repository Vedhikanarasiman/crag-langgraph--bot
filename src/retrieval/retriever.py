from src.embeddings.embedder import load_vectorstore

def retrieve_chunks(query: str, k: int = 5) -> list[dict]:
    """
    Takes a user question
    Searches ChromaDB for top-k most relevant chunks
    Returns list of chunks with text + metadata
    """

    # Load the vectorstore from disk
    vectorstore = load_vectorstore()

    # Search for top k similar chunks
    results = vectorstore.similarity_search(query, k=k)

    # Format results nicely
    chunks = []
    for doc in results:
        chunks.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    return chunks