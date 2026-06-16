from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Takes list of page dicts from loader.py
    Splits each page's text into smaller overlapping chunks.
    Each chunk keeps the original metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # max characters per chunk
        chunk_overlap=50,      # overlap between chunks
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []

    for doc in documents:
        # Split this page's text into chunks
        split_texts = splitter.split_text(doc["text"])

        # Each chunk gets the same metadata as its parent page
        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page"],
                    "chunk_index": i
                }
            })

    print(f"Total chunks created: {len(chunks)}")
    return chunks