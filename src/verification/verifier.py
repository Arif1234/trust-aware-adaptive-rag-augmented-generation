from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def verify_claims(
        claims,
        retrieved_docs
):

    evidence = " ".join(

        doc["content"]
        if isinstance(doc, dict)
        else doc

        for doc in retrieved_docs
    )

    e_emb = embed_model.encode(
        [evidence]
    )

    verified = []

    for claim in claims:

        c_emb = embed_model.encode(
            [claim]
        )

        score = cosine_similarity(
            c_emb,
            e_emb
        )[0][0]

        verified.append({
            "claim": claim,
            "score": float(score),
            "hallucinated": score < 0.50
        })

    return verified