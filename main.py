# clovatar_backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from generate_prompt import build_gpt_prompt
from pinecone_utils import query_relevant_chunks

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    user_input: str
    creator_id: str
    session_id: str

@app.post("/chat")
async def chat_with_creator(data: ChatInput):
    print("\n⚡ Incoming request data:", data.dict())

    # Query Pinecone for context and links
    chunks, product_links = query_relevant_chunks(data.user_input, data.creator_id)

    print(f"📄 Retrieved {len(chunks)} context chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk[:100]}...")  # Print first 100 chars

    print(f"🛍️ Retrieved {len(product_links)} product links:")
    for i, link in enumerate(product_links, 1):
        print(f"  {i}. {link}")

    # Build GPT prompt
    prompt = build_gpt_prompt(
        user_input=data.user_input,
        context_chunks=chunks,
        creator_id=data.creator_id,
        product_links=product_links,
        session_id=data.session_id
    )

    print("\n🧠 Final GPT Prompt:\n", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": data.user_input}
            ],
            temperature=0.7
        )
        reply = response.choices[0].message.content
        print("\n✅ GPT Reply:\n", reply)
    except Exception as e:
        reply = f"Error from GPT: {str(e)}"
        print("\n❌ GPT Error:\n", reply)

    return {"reply": reply}
