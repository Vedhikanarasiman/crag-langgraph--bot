from src.ingestion.loader import load_pdfs
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import store_embeddings

def main():
    print("ResearchMind Ingestion Pipeline\n")

    # Step 1: Load PDFs
    print("Step 1: Loading PDFs")
    documents = load_pdfs("data/papers")

    # Step 2: Chunk documents
    print("\nStep 2: Chunking documents")
    chunks = chunk_documents(documents)

    # Step 3: Embed and store
    print("\nStep 3: Embedding and storing in ChromaDB")
    store_embeddings(chunks)

    print("\n Ingestion Complete! ")

if __name__ == "__main__":
    main()