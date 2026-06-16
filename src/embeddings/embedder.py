from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os

# Path where ChromaDB will save vectors on disk
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "researchmind"

def get_embeddings():
    """
    Returns the Ollama embedding model.
    This is what converts text → vectors.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text", 
        base_url="http://127.0.0.1:11434"
    )  # force correct port

def store_embeddings(chunks: list[dict]):
    """
    Takes chunks from chunker.py
    Converts each chunk to a vector
    Stores vectors + text + metadata in ChromaDB
    """
    print("Generating embeddings and storing in ChromaDB.")

    # Separate texts and metadatas for ChromaDB
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Create and persist ChromaDB collection
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print(f"Stored {len(texts)} chunks in ChromaDB at ./{CHROMA_PATH}/")
    return vectorstore

def load_vectorstore():
    """
    Loads existing ChromaDB from disk.
    Used at query time — no need to re-embed.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH
    )