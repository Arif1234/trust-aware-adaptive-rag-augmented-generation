import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.verification.claim_extractor import extract_claims
from src.verification.verifier import verify_claims


def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            rows.append(
                json.loads(line)
            )

    return rows


def evaluate_response(item):

    response_text = item.get(
        "response",
        ""
    )

    labels = item.get(
        "labels",
        []
    )

    actual_label = (
        1 if len(labels) > 0 else 0
    )

    claims = extract_claims(
        response_text
    )

    # Use source context if available
    source_text = item.get(
        "source_info",
        ""
    )

    retrieved_docs = [
        {
            "source": "RAGTruth",
            "content": source_text
        }
    ]

    verification = verify_claims(
        claims,
        retrieved_docs
    )

    hallucinated_claims = sum(
        1
        for item in verification
        if item["hallucinated"]
    )

    if len(claims) == 0:
        predicted_label = 0

    else:

        hallucination_ratio = (
            hallucinated_claims
            / len(claims)
        )

        predicted_label = (
            1
            if hallucination_ratio >= 0.5
            else 0
        )

    return {

        "actual_label":
            actual_label,

        "predicted_label":
            predicted_label,

        "claim_count":
            len(claims),

        "hallucinated_claims":
            hallucinated_claims,

        "hallucination_ratio":
        (
            hallucinated_claims / len(claims)
            if len(claims) > 0
            else 0
        )
    }


def evaluate_dataset(data):

    y_true = []

    y_pred = []

    results = []

    for idx, item in enumerate(
        data,
        start=1
    ):

        try:

            result = evaluate_response(
                item
            )

            y_true.append(
                result["actual_label"]
            )

            y_pred.append(
                result["predicted_label"]
            )

            results.append(
                result
            )

            if idx % 100 == 0:

                print(
                    f"Processed {idx}"
                )

        except Exception as e:

            print(
                f"Skipping {idx}: {e}"
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

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    summary = {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "total_records":
            len(y_true),

        "confusion_matrix":
            cm.tolist()
    }

    return summary, results


if __name__ == "__main__":

    response_data = load_jsonl(
        "data/response.jsonl"
    )

    print(
        f"Loaded {len(response_data)} responses"
    )

    summary, results = evaluate_dataset(
        response_data
    )

    print("\n")
    print("=" * 70)
    print("RAGTRUTH EVALUATION")
    print("=" * 70)

    for k, v in summary.items():

        print(f"{k}: {v}")

    output = {

        "summary": summary,
        "results": results
    }

    with open(
        "ragtruth_evaluation.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print(
        "\nResults saved to "
        "ragtruth_evaluation.json"
    )