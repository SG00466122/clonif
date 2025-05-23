# clovatar_backend/upload_to_pinecone.py

import json
import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def embed_text(text):
    clean_text = text.strip()
    if not clean_text:
        return None
    response = client.embeddings.create(
        input=clean_text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def chunk_text(text, max_tokens=300):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def upload_json_to_pinecone(json_path, creator_id):
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    to_upsert = []
    for entry in entries:
        text_chunks = chunk_text(entry["transcription"])
        for chunk in text_chunks:
            vector = embed_text(chunk)
            if vector is None:
                continue
            metadata = {
                "creator_id": creator_id,
                "title": entry.get("title", ""),
                "date": entry.get("date", ""),
                "tags": entry.get("tags", []),
                "topic_tags": entry.get("topic_tags", []),
                "product_links": json.dumps(entry.get("product_links", [])),
                "text": chunk
            }
            to_upsert.append((str(uuid.uuid4()), vector, metadata))

    index.upsert(vectors=to_upsert, namespace=creator_id)
    print(f"✅ Uploaded {len(to_upsert)} chunks to Pinecone for creator '{creator_id}'")

# Example:
upload_json_to_pinecone("data/fittuber.json", "fittuber")
