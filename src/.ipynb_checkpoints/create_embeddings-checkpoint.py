from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = []

# build from source_info
for item in sources:

    docs = item["source_info"]

    if isinstance(docs, list):
        documents.extend(docs)
    else:
        documents.append(docs)

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