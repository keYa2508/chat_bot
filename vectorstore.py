import chromadb
from typing import List
import os

# Use persistent storage so your data is saved to disk
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection(namespace: str = os.getenv("name_space")):
    """
    In Pinecone you used `namespace`.
    In ChromaDB we’ll map `namespace` -> `collection name`.
    """
    return chroma_client.get_or_create_collection(name=namespace)

def store_in_chromadb(
    chunks: List[str],
    embeddings: List[List[float]],
    namespace: str = os.getenv("name_space")
):
    collection = get_collection(namespace)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    # You can also store the full text as `documents`
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks,
    )

def search_in_chromadb(query_vector: List[float], top_k: int = 5, namespace: str = os.getenv("name_space")):
    collection = get_collection(namespace)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    return results
