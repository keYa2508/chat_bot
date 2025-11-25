from ollama import Client
import os

client = Client(host=os.getenv("ollama_host"))
model = os.getenv("llm_model")

def query_llm_with_context(query: str, context: str):
    system_context = (
        "You are an expert assistant. Use the following context to answer the query:\n\n"
        f"{context}"
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_context},
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext:\n{context}",
            },
        ],
        options={
            "temperature": 0.2,
        },
    )

    return response["message"]["content"]
