import os
import sqlite3
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Init Pinecone (modern SDK)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "bot"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# SQLite setup
DB_PATH = "agriculture.db"
TABLES = [
    "cropbooking_by_rbks_2024_dashboard",
    "day_ext_irr_method_2024_dashboard",
    "cr_dist_top25crops_mv",
    "cr_authdetails_vill_mvr_2024"
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_rows_as_text():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    docs = []

    for table in TABLES:
        cur.execute(f"SELECT * FROM {table}")
        cols = [desc[0] for desc in cur.description]
        for row in cur.fetchall():
            text = ", ".join([f"{cols[i]}: {row[i]}" for i in range(len(cols))])
            docs.append(text)

    conn.close()
    return docs

def upload_to_pinecone():
    texts = fetch_rows_as_text()
    print(f"📄 Embedding {len(texts)} rows")
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeds = model.encode(batch).tolist()
        records = [
            {
                "id": str(uuid.uuid4()),
                "values": embeds[j],
                "metadata": {"text": batch[j]}
            }
            for j in range(len(batch))
        ]
        index.upsert(vectors=records)
    print("✅ Uploaded to Pinecone")

if __name__ == "__main__":
    upload_to_pinecone()
