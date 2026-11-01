from .load_data import sources
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import json

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

documents = []

# build from source_info
for item in sources:

    docs = to_text(item["source_info"])

    if isinstance(docs, list):
        documents.extend([to_text(doc) for doc in docs])
    else:
        documents.append(to_text(docs))

documents = [doc for doc in documents if doc.strip()]

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

faiss.write_index(
    index,
    "faiss_store/ragtruth.index"
)

pickle.dump(
    documents,
    open(
        "faiss_store/docs.pkl",
        "wb"
    )
)