from ollama import Client
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()
client = Client(host = os.getenv("ollama_host"))
EMBEDDED_MODEL = os.getenv("embeded_modal")

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        response = client.embeddings(
            model=EMBEDDED_MODEL,
            prompt=chunk
        )

        embeddings.append(response.embedding)
    return embeddings

def embed_user_query(query: str) -> List[float]:
    response = client.embeddings(
        model=EMBEDDED_MODEL,
        prompt=query
    )
    return response.embedding