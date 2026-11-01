from src.retrieval.adaptive_retrieval import retrieve
from src.retrieval.trust_scoring import compute_trust

from src.generation.llama_rag import generate_answer

from src.verification.claim_extractor import (
    extract_claims
)

from src.verification.verifier import (
    verify_claims
)


# =======================================
# MODEL A
# BASELINE RAG
# =======================================

def run_model_a(
        question,
        documents
):

    context = "\n\n".join(
        doc["content"]
        for doc in documents[:3]
    )

    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer,
        "hallucinated": False
    }


# =======================================
# MODEL B
# ADAPTIVE RETRIEVAL
# =======================================

def run_model_b(question):

    documents = retrieve(
        question
    )

    context = "\n\n".join(
        doc["content"]
        for doc in documents[:3]
    )

    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer,
        "hallucinated": False
    }


# =======================================
# MODEL C
# ADAPTIVE + TRUST
# =======================================

def run_model_c(question):

    documents = retrieve(
        question
    )

    trust_score = compute_trust(
        documents,
        query=question,
        source_type="academic"
    )

    if trust_score < 0.50:

        return {
            "answer": "",
            "hallucinated": True
        }

    context = "\n\n".join(
        doc["content"]
        for doc in documents[:3]
    )

    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer,
        "hallucinated": False
    }


# =======================================
# MODEL D
# PROPOSED FRAMEWORK
# =======================================

def run_model_d(question):

    documents = retrieve(
        question
    )

    trust_score = compute_trust(
        documents,
        query=question,
        source_type="academic"
    )

    if trust_score < 0.50:

        return {
            "answer": "",
            "hallucinated": True
        }

    context = "\n\n".join(
        doc["content"]
        for doc in documents[:3]
    )

    answer = generate_answer(
        question,
        context
    )

    claims = extract_claims(
        answer
    )

    verification = verify_claims(
        claims,
        documents
    )

    hallucinated_claims = sum(
        1
        for item in verification
        if item["hallucinated"]
    )

    if len(claims) == 0:

        predicted = False

    else:

        predicted = (

            hallucinated_claims
            / len(claims)

        ) >= 0.75

    return {
        "answer": answer,
        "hallucinated": predicted
    }