import matplotlib.pyplot as plt
import numpy as np
import os


def generate_class_distribution():
    """
    Generate class distribution visualization for RAGTruth evaluation sample.
    
    Statistics:
    - Non-hallucinated: 131 (65.5%)
    - Hallucinated: 69 (34.5%)
    - Total: 200
    """
    
    # Data from evaluation results
    non_hallucinated = 131
    hallucinated = 69
    total = 200
    
    labels = ['Non-hallucinated', 'Hallucinated']
    values = [non_hallucinated, hallucinated]
    percentages = [65.5, 34.5]
    colors = ['#2ecc71', '#e74c3c']  # Green for non-hallucinated, red for hallucinated
    
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # OPTION 1: DONUT CHART (Recommended for thesis)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'weight': 'bold'},
        pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Create the donut hole
    centre_circle = plt.Circle(
        (0, 0),
        0.70,
        fc='white',
        linewidth=2,
        edgecolor='#999999'
    )
    ax.add_artist(centre_circle)
    
    # Customize percentage text (white, inside donut)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_weight('bold')
    
    # Customize label text
    for text in texts:
        text.set_fontsize(13)
        text.set_weight('bold')
    
    ax.set_title(
        'Class Distribution of the 200-Record RAGTruth Evaluation Sample',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add legend with counts and percentages
    legend_labels = [
        f'Non-hallucinated: {values[0]} ({percentages[0]}%)',
        f'Hallucinated: {values[1]} ({percentages[1]}%)'
    ]
    ax.legend(
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    
    # Save the donut chart
    donut_path = os.path.join(OUTPUT_DIR, 'class_distribution_donut.png')
    plt.savefig(donut_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Donut chart saved: {donut_path}")
    
    # =========================================================================
    # OPTION 2: HORIZONTAL BAR CHART (Alternative)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos,
        values,
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=2
    )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Responses', fontsize=13, fontweight='bold')
    
    ax.set_title(
        'Class Distribution of the 200-Record RAGTruth Evaluation Sample',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add value labels on bars
    for i, (v, p) in enumerate(zip(values, percentages)):
        ax.text(
            v + 2,
            i,
            f'{v} ({p}%)',
            va='center',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlim(0, max(values) + 20)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    # Save the bar chart
    bar_path = os.path.join(OUTPUT_DIR, 'class_distribution_bar.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bar chart saved: {bar_path}")
    
    # =========================================================================
    # PRINT STATISTICS
    # =========================================================================
    print("\n" + "="*60)
    print("RAGTruth EVALUATION SAMPLE CLASS DISTRIBUTION")
    print("="*60)
    print(f"Non-hallucinated responses: {values[0]:>3} ({percentages[0]:>5.1f}%)")
    print(f"Hallucinated responses:     {values[1]:>3} ({percentages[1]:>5.1f}%)")
    print(f"{'─'*60}")
    print(f"Total responses:            {total:>3} (100.0%)")
    print("="*60 + "\n")
    
    return donut_path, bar_path


if __name__ == "__main__":
    generate_class_distribution()
