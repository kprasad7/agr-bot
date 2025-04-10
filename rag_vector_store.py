import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("bot")  # Pinecone index name

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_vector_context_from_query(query: str, k: int = 3) -> str:
    query_vector = model.encode(query).tolist()
    response = index.query(vector=query_vector, top_k=k, include_metadata=True)

    matches = response.get("matches", [])
    if not matches:
        return "No relevant results."

    return "\n".join([match["metadata"].get("text", "") for match in matches])
