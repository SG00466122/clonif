# clovatar_backend/generate_prompt.py

def build_gpt_prompt(user_input, context_chunks, creator_id, product_links, session_id):
    tone_presets = {
        "geekyranjit": "You are GeekyRanjit, an honest, practical Indian tech reviewer. You speak in a calm, informative tone. You explain clearly without over-selling. Recommend links only when truly relevant, and mention them like a helpful friend.",
        "default": "You are a knowledgeable, helpful expert who replies in a friendly and human way."
    }

    tone = tone_presets.get(creator_id, tone_presets["default"])
    context = "\n\n".join(context_chunks)

    links_text = ""
    if product_links:
        links_text += "\nHere are some helpful things related to this question:\n"
        for item in product_links:
            links_text += f"- **{item['name']}** → {item['description']} [Visit Here]({item['link']})\n"

    prompt = f"""{tone}

Context:
\"\"\"
{context}
\"\"\"

Current user message:
\"{user_input}\"

If relevant, provide clear and friendly information, and include links like you're sharing something genuinely helpful. Avoid sounding robotic or pushy.

{links_text}
"""

    return prompt
