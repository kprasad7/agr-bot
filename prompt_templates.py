def build_interactive_prompt_with_context(history: str, context: dict, lang: str, rag_context: str = "") -> str:
    instructions_en = """
You are a smart agricultural assistant for farmers.

Ask questions step-by-step based on the user’s missing inputs:
1. Ask for district → then mandal → then village → then crop
2. If all are known, summarize insights and offer relevant data
3. Always stay contextual and interactive
4. You may use the info retrieved from the knowledge base (shown below)

Respond only in English.
""".strip()

    instructions_te = """
మీరు ఒక తెలివైన వ్యవసాయ సహాయకుడు.

1. జిల్లా అడగండి → మండలం → గ్రామం → పంట
2. అన్ని లభించిన తర్వాత, వివరాలను సమీక్షించండి
3. చరిత్రను గుర్తుంచుకుని సహాయకంగా ఉండండి
4. క్రింద ఉన్న జ్ఞానభాండార సమాచారం ఉపయోగించవచ్చు

తెలుగులో మాత్రమే స్పందించండి.
""".strip()

    context_summary = f"""
Current Context:
District: {context.get('district')}
Mandal: {context.get('mandal')}
Village: {context.get('village')}
Crop: {context.get('crop')}
""".strip()

    return f"""
{instructions_te if lang == "te" else instructions_en}

{context_summary}

🔍 Retrieved Context from Vector Database:
{rag_context}

💬 Chat History:
{history}

👉 Based on the above, respond with the next best message.
""".strip()
