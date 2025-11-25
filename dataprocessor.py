from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_chromadb
import os

pdf_path = os.getenv("file_path")

def run():
    pages = read_pdf(pdf_path)

    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)

    embedding = embed_chunks(chunks)

    store_in_chromadb(chunks, embedding, namespace=os.getenv("name_space"))

    
if __name__ == "__main__":
    run()