from embedder import embed_user_query
from vectorstore import search_in_chromadb
from llm import query_llm_with_context
import os

def process_user_query(query: str):
    query_vector = embed_user_query(query)

    match_chunks = search_in_chromadb(query_vector, top_k=5, namespace=os.getenv("name_space"))

    generated_response = query_llm_with_context(query, match_chunks)
    return generated_response

if __name__ == "__main__":
    user_query= "Explain the key contributions of Karthikeyan Manikandan in the field of robotics."
    process_user_query(user_query)