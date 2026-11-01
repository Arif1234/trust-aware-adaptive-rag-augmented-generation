import json
import os
import matplotlib.pyplot as plt
import numpy as np


def generate_confusion_matrix():
    """Generate confusion matrix visualization with TP, FP, TN, FN labels"""
    
    try:
        # Load confusion matrix from RAGTruth evaluation
        with open("ragtruth_evaluation.json", "r", encoding="utf-8") as f:
            ragtruth_data = json.load(f)
            cm = np.array(
                ragtruth_data["summary"]["confusion_matrix"]
            )
    except FileNotFoundError:
        print("Error: ragtruth_evaluation.json not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Display confusion matrix heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(
        ['Negative (0)', 'Positive (1)'],
        fontsize=11
    )
    ax.set_yticklabels(
        ['Negative (0)', 'Positive (1)'],
        fontsize=11
    )
    
    ax.set_ylabel(
        'True Label',
        fontsize=12,
        fontweight='bold'
    )
    ax.set_xlabel(
        'Predicted Label',
        fontsize=12,
        fontweight='bold'
    )
    
    # Extract confusion matrix values
    tn = int(cm[0, 0])  # True Negatives
    fp = int(cm[0, 1])  # False Positives
    fn = int(cm[1, 0])  # False Negatives
    tp = int(cm[1, 1])  # True Positives
    
    # Create annotations
    annotations = [
        [f'TN\n{tn}', f'FP\n{fp}'],
        [f'FN\n{fn}', f'TP\n{tp}']
    ]
    
    # Add text annotations
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = (
                "white" if cm[i, j] > threshold
                else "black"
            )
            plt.text(
                j, i,
                annotations[i][j],
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight='bold'
            )
    
    # Set title
    plt.title(
        'Confusion Matrix - Hallucination Detection\n'
        '(True Negatives, False Positives, '
        'False Negatives, True Positives)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    
    # Save figure
    plt.savefig(
        "results/confusion_matrix.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    
    # Print metrics to console
    print("\n" + "="*60)
    print("CONFUSION MATRIX METRICS")
    print("="*60)
    print(f"True Negatives (TN):  {tn:>10,}")
    print(f"False Positives (FP): {fp:>10,}")
    print(f"False Negatives (FN): {fn:>10,}")
    print(f"True Positives (TP):  {tp:>10,}")
    print("="*60)
    print(f"\nConfusion matrix saved: results/confusion_matrix.png")
    
    return tn, fp, fn, tp


if __name__ == "__main__":
    generate_confusion_matrix()