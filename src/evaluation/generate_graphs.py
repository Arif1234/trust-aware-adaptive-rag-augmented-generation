import json
import os

import matplotlib.pyplot as plt
import numpy as np


RESULT_FILE = "results/model_comparison.json"

OUTPUT_DIR = "results"


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


models = list(results.keys())

accuracy = [
    results[m]["accuracy"]
    for m in models
]

precision = [
    results[m]["precision"]
    for m in models
]

recall = [
    results[m]["recall"]
    for m in models
]

f1 = [
    results[m]["f1"]
    for m in models
]

latency = [
    results[m]["latency_seconds"]
    for m in models
]


def save_bar_chart(
        values,
        title,
        ylabel,
        filename
):

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        models,
        values
    )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.xticks(
        rotation=15
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

    plt.close()


save_bar_chart(
    accuracy,
    "Model Accuracy Comparison",
    "Accuracy",
    "accuracy_comparison.png"
)

save_bar_chart(
    precision,
    "Model Precision Comparison",
    "Precision",
    "precision_comparison.png"
)

save_bar_chart(
    recall,
    "Model Recall Comparison",
    "Recall",
    "recall_comparison.png"
)

save_bar_chart(
    f1,
    "Model F1 Comparison",
    "F1 Score",
    "f1_comparison.png"
)

save_bar_chart(
    latency,
    "Model Latency Comparison",
    "Seconds",
    "latency_comparison.png"
)

# =======================================
# COMBINED METRICS GRAPH
# =======================================

plt.figure(figsize=(12, 6))

x = np.arange(len(models))
width = 0.25

bars1 = plt.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

bars2 = plt.bar(
    x,
    recall,
    width,
    label="Recall"
)

bars3 = plt.bar(
    x + width,
    f1,
    width,
    label="F1"
)

for bars in [bars1, bars2, bars3]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

plt.xticks(
    x,
    models,
    rotation=15
)

plt.ylabel("Score")

plt.title(
    "Precision, Recall and F1 Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "combined_metrics.png"
    )
)

plt.close()

# =======================================
# HALLUCINATION DETECTION
# =======================================

hallucination_detection = [
    1 - acc
    for acc in accuracy
]

save_bar_chart(
    hallucination_detection,
    "Hallucination Detection Capability",
    "Detection Rate",
    "hallucination_detection.png"
)

# =======================================
# LATENCY VS F1
# =======================================

plt.figure(figsize=(10, 6))

plt.scatter(
    latency,
    f1,
    s=150
)

for i, model in enumerate(models):

    plt.annotate(
        model,
        (
            latency[i],
            f1[i]
        )
    )

plt.xlabel(
    "Latency (seconds)"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "Latency vs F1 Tradeoff"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "latency_vs_f1.png"
    )
)

plt.close()

# =======================================
# ABLATION STUDY
# =======================================

ablation_scores = f1

save_bar_chart(
    ablation_scores,
    "Ablation Study (F1 Score)",
    "F1 Score",
    "ablation_study.png"
)

print(
    "\nGraphs generated successfully."
)

print(
    f"Saved in: {OUTPUT_DIR}"
)