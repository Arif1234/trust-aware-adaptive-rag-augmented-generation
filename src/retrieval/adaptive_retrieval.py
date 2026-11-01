import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "faiss_store/rag.index"
)

documents = pickle.load(
    open(
        "faiss_store/docs.pkl",
        "rb"
    )
)


def retrieve(query):

    q_emb = embed_model.encode(
        [query]
    )

    D, I = index.search(
        np.array(q_emb).astype("float32"),
        10
    )

    avg_score = 1/(1+D[0].mean())

    if avg_score > 0.80:
        k = 3

    elif avg_score > 0.60:
        k = 5

    else:
        k = 10

    results = []

    for idx in I[0][:k]:
        results.append(
            {
                "source": documents[idx]["source"],
                "content": documents[idx]["content"]
            }
        )

    return results