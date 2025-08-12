import requests
import pandas as pd

url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')


import os
os.environ["TQDM_NOTEBOOK"] = "0"   # set before importing tqdm
from tqdm.auto import tqdm

from tqdm.auto import tqdm
from tqdm import TqdmWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmWarning)



# Q4: Qdrant + Jina v2-small-en (question + answer), limit=5 → MRR
# pip install qdrant-client sentence-transformers

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from tqdm import tqdm
import numpy as np

# --- 1) Prep embeddings (question + answer) ---
model_handle = "jinaai/jina-embeddings-v2-small-en"
model = SentenceTransformer(model_handle, trust_remote_code=True)  # HF model card requires trust_remote_code
# This model outputs 512-d vectors.  ✔️ :contentReference[oaicite:0]{index=0}

texts_qA = [f"{d['question']} {d['text']}" for d in documents]
embs = model.encode(texts_qA, normalize_embeddings=True)

# --- 2) (Re)create a Qdrant collection with cosine distance and the correct vector size (512) ---
client = QdrantClient(host="localhost", port=6333)  # adjust if needed
COLL = "qa_qdrant_jina_small"

client.recreate_collection(
    collection_name=COLL,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
# Qdrant uses cosine as dot product over normalized vectors; it normalizes vectors internally. :contentReference[oaicite:1]{index=1}

# Optional but recommended if you filter by `course`: create a payload index for faster filtered search. :contentReference[oaicite:2]{index=2}
# client.create_payload_index(COLL, field_name="course", field_schema="keyword")

# --- 3) Upsert points (store your doc id as the Qdrant point id) ---
points = [
    PointStruct(
        id=str(doc["id"]),                           # keep types consistent; ground_truth may be str/int
        vector=embs[i].tolist(),
        payload={
            "id": str(doc["id"]),
            "question": doc["question"],
            "text": doc["text"],
            "course": doc["course"],
        },
    )
    for i, doc in enumerate(documents)
]
client.upsert(collection_name=COLL, points=points)

# --- 4) Search helper: encode query, filter by course, limit=5 ---
def qdrant_search(question, course, limit=5):
    qvec = model.encode([question], normalize_embeddings=True)[0]
    flt = Filter(must=[FieldCondition(key="course", match=MatchValue(value=course))])
    hits = client.search(
        collection_name=COLL,
        query_vector=qvec.tolist(),
        query_filter=flt,
        with_payload=True,
        limit=limit,  # top-5 results
    )
    return hits  # list[ScoredPoint] with .id, .score, .payload  :contentReference[oaicite:3]{index=3}

# --- 5) Compute MRR ---
def mrr_for_qdrant(ground_truth, limit=5):
    total = 0.0
    for q in tqdm(ground_truth):
        gold = str(q["document"])
        hits = qdrant_search(q["question"], q["course"], limit=limit)
        rank = None
        for i, h in enumerate(hits):
            if str(h.id) == gold:   # compare IDs to IDs
                rank = i + 1
                break
        total += 0.0 if rank is None else 1.0 / rank
    return total / len(ground_truth)

mrr_value = mrr_for_qdrant(ground_truth, limit=5)
print("MRR (Qdrant, Jina v2-small-en, Q+A, top-5):", mrr_value)
