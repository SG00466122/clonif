# clovatar_backend/pinecone_utils.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def embed_text(text):
    clean_text = text.strip()
    if not clean_text:
        return None
    response = client.embeddings.create(
        input=clean_text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def query_relevant_chunks(user_input, creator_id, top_k=5):
    user_vector = embed_text(user_input)
    if user_vector is None:
        return [], []

    response = index.query(
        vector=user_vector,
        top_k=top_k,
        namespace=creator_id,
        include_metadata=True
    )

    chunks = []
    product_links = []

    for match in response.get("matches", []):
        metadata = match["metadata"]
        chunks.append(metadata.get("text", ""))
        if "product_links" in metadata:
            try:
                product_links.extend(eval(metadata["product_links"]))
            except:
                continue

    return chunks, product_links
