from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

SOURCE_WEIGHTS = {

    "government": 1.0,
    "academic": 0.95,
    "wiki": 0.80,
    "blog": 0.50,
    "unknown": 0.30
}


def compute_trust(
        retrieved_docs,
        source_type="unknown"
):

    embs = embed_model.encode(
        retrieved_docs
    )

    consistency = cosine_similarity(
        embs
    ).mean()

    reliability = SOURCE_WEIGHTS.get(
        source_type,
        0.30
    )

    trust = (
            0.4 * reliability
            +
            0.6 * consistency
    )

    return trust