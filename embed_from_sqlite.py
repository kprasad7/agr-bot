import sqlite3
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "agri.db"
VECTOR_FOLDER = "vector_db"
TABLES = [
    "cropbooking_by_rbks_2024_dashboard",
    "day_ext_irr_method_2024_dashboard",
    "cr_dist_top25crops_mv",
    "cr_authdetails_vill_mvr_2024"
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_rows(cursor, table):
    cursor.execute(f"SELECT * FROM {table}")
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return [", ".join([f"{columns[i]}: {str(val)}" for i, val in enumerate(row)]) for row in rows]

def get_all_texts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    all_rows = []
    for table in TABLES:
        all_rows.extend(extract_text_rows(cursor, table))
    conn.close()
    return all_rows

def build_and_save_index():
    docs = get_all_texts()
    print(f"✅ Total records: {len(docs)}")
    embeddings = model.encode(docs, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(VECTOR_FOLDER, exist_ok=True)
    faiss.write_index(index, f"{VECTOR_FOLDER}/faiss.index")
    with open(f"{VECTOR_FOLDER}/index.pkl", "wb") as f:
        pickle.dump({"docs": docs}, f)

    print("✅ FAISS index saved.")

if __name__ == "__main__":
    build_and_save_index()


