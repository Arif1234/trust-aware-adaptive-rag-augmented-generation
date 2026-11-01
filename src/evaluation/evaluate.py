import json
import time

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from src.retrieval.adaptive_retrieval import retrieve
from src.retrieval.trust_scoring import compute_trust
from src.generation.llama_rag import generate_answer

from src.verification.claim_extractor import extract_claims
from src.verification.verifier import verify_claims


def load_test_questions(file_path):

    questions = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            questions.append(item)

    return questions


def evaluate_question(question):

    start_time = time.time()

    documents = retrieve(question)

    trust_score = compute_trust(
        documents,
        source_type="academic"
    )

    context = "\n\n".join(
        doc[:1000]
        for doc in documents[:3]
    )

    answer = generate_answer(
        question,
        context
    )

    claims = extract_claims(answer)

    verification = verify_claims(
        claims,
        documents
    )

    latency = (
        time.time() - start_time
    )

    total_claims = len(verification)

    hallucinated_claims = sum(
        item["hallucinated"]
        for item in verification
    )

    hallucination_rate = 0

    if total_claims > 0:

        hallucination_rate = (
            hallucinated_claims
            / total_claims
        )

    return {
        "question": question,
        "answer": answer,
        "trust_score": float(trust_score),
        "total_claims": int(total_claims),
        "hallucinated_claims": int(hallucinated_claims),
        "hallucination_rate": float(hallucination_rate),
        "latency": float(latency)
    }


def evaluate_dataset(test_questions):

    trust_scores = []

    hallucination_rates = []

    latencies = []

    y_true = []
    y_pred = []

    results = []

    for idx, question in enumerate(
        test_questions,
        start=1
    ):

        print(
            f"\nEvaluating "
            f"{idx}/{len(test_questions)}"
        )

        result = evaluate_question(
            question
        )

        results.append(result)

        trust_scores.append(
            result["trust_score"]
        )

        hallucination_rates.append(
            result["hallucination_rate"]
        )

        latencies.append(
            result["latency"]
        )

        # Placeholder labels
        # Replace with RAGTruth labels later

        actual_label = 0

        predicted_label = (
            1
            if result[
                "hallucination_rate"
            ] > 0
            else 0
        )

        y_true.append(
            actual_label
        )

        y_pred.append(
            predicted_label
        )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    summary = {

        "avg_trust_score":
            sum(trust_scores)
            / len(trust_scores),

        "avg_hallucination_rate":
            sum(hallucination_rates)
            / len(hallucination_rates),

        "avg_latency":
            sum(latencies)
            / len(latencies),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1

    }

    return summary, results

if __name__ == "__main__":

    test_questions = [

        "What is retrieval augmented generation?",

        "What is LangChain?",

        "What is FAISS?",

        "What is vector search?",

        "What are embeddings?"

    ]

    summary, results = evaluate_dataset(
        test_questions
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for k, v in summary.items():

        print(
            f"{k}: {v:.4f}"
            if isinstance(v, float)
            else f"{k}: {v}"
        )

with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )