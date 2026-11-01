print(
    "evaluate_models.py loaded"
)

import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.retrieval.adaptive_retrieval import retrieve
from src.retrieval.trust_scoring import compute_trust

from src.generation.llama_rag import generate_answer
from src.evaluation.model_runners import (
    run_model_a,
    run_model_b,
    run_model_c,
    run_model_d
)

from src.verification.claim_extractor import (
    extract_claims
)

from src.verification.verifier import (
    verify_claims
)

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

def evaluate_model(
        model_name,
        responses,
        source_map
):

    print("\n" + "=" * 60)
    print(f"STARTING MODEL {model_name}")
    print("=" * 60)

    y_true = []
    y_pred = []

    TEST_SIZE = 200

    start_time = time.time()

    for idx, item in enumerate(
            responses[:TEST_SIZE],
            start=1
    ):

        print(
            f"Model {model_name}: "
            f"{idx}/{TEST_SIZE}"
        )

        question = ""

        source_id = item.get(
            "source_id"
        )

        if source_id in source_map:

            question = source_map[
                source_id
            ].get(
                "prompt",
                ""
            )

        actual_label = (

            1

            if len(
                item.get(
                    "labels",
                    []
                )
            ) > 0

            else 0

        )

        try:

            if model_name == "A":

                retrieved_docs = retrieve(
                    question
                )

                result = run_model_a(
                    question,
                    retrieved_docs
                )

            elif model_name == "B":

                result = run_model_b(
                    question
                )

            elif model_name == "C":

                result = run_model_c(
                    question
                )

            else:

                result = run_model_d(
                    question
                )

            predicted_label = (

                1

                if result[
                    "hallucinated"
                ]

                else 0

            )

            y_true.append(
                actual_label
            )

            y_pred.append(
                predicted_label
            )

        except Exception as e:

            print(
                f"\nERROR "
                f"Model {model_name} "
                f"Record {idx}"
            )

            print(e)

            continue

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nCompleted Model {model_name}"
    )

    print(
        f"Processed: {len(y_true)}"
    )

    print(
        f"Elapsed: {elapsed:.2f}s"
    )

    if len(y_true) == 0:

        return {

            "accuracy": 0,

            "precision": 0,

            "recall": 0,

            "f1": 0,

            "latency_seconds":
                elapsed

        }

    result = {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "latency_seconds":
            elapsed

    }

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    return result

# =======================================
# MAIN
# =======================================

if __name__ == "__main__":

    print("=" * 70)
    print("EVALUATE MODELS STARTED")
    print("=" * 70)

    responses = load_jsonl(
        "data/response.jsonl"
    )

    print(
        f"Loaded Responses: {len(responses)}"
    )

    sources = load_jsonl(
        "data/source_info.jsonl"
    )

    print(
        f"Loaded Sources: {len(sources)}"
    )

    source_map = {

        item["source_id"]: item

        for item in sources
    }

    print("\nStarting Model Comparison...\n")

    results = {

        "Model_A_Baseline":
            evaluate_model(
                "A",
                responses,
                source_map
            ),

        "Model_B_Adaptive":
            evaluate_model(
                "B",
                responses,
                source_map
            ),

        "Model_C_Adaptive_Trust":
            evaluate_model(
                "C",
                responses,
                source_map
            ),

        "Model_D_Proposed":
            evaluate_model(
                "D",
                responses,
                source_map
            )

    }

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        json.dumps(
            results,
            indent=4
        )
    )

    with open(
        "results/model_comparison.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nSaved to results/model_comparison.json"
    )