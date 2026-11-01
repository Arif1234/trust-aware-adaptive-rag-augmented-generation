from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        query,
        source_type="unknown"
):

    texts = []

    for doc in retrieved_docs:

        if isinstance(doc, dict):
            texts.append(
                doc["content"]
            )

        else:
            texts.append(doc)

    embs = embed_model.encode(
        texts
    )

    query_emb = embed_model.encode(
        [query]
    )

    relevance = cosine_similarity(
        query_emb,
        embs
    ).mean()

    consistency = cosine_similarity(
        embs
    ).mean()

    reliability = SOURCE_WEIGHTS.get(
        source_type,
        0.30
    )

    trust = (
            0.4 * relevance
            + 0.3 * reliability
            + 0.3 * consistency
    )

    return float(trust)