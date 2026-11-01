import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Load model comparison data
with open("results/model_comparison.json", "r") as f:
    data = json.load(f)

# Extract data
models = ['Model A\n(Baseline)', 'Model B\n(Adaptive)', 'Model C\n(Adaptive+Trust)', 'Model D\n(Proposed)']
model_keys = list(data.keys())

accuracy = [data[key]['accuracy'] for key in model_keys]
precision = [data[key]['precision'] for key in model_keys]
recall = [data[key]['recall'] for key in model_keys]
f1 = [data[key]['f1'] for key in model_keys]

# Create figure
fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

# Set up bar positions
x = np.arange(len(models))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3498db', edgecolor='black', linewidth=1)
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#e74c3c', edgecolor='black', linewidth=1)
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#2ecc71', edgecolor='black', linewidth=1)
bars4 = ax.bar(x + 1.5*width, f1, width, label='F1', color='#f39c12', edgecolor='black', linewidth=1)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

# Customize plot
ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Figure 5.x: Combined Classification Performance Across the Four Model Configurations', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add performance note
performance_text = "Model D (Proposed) achieves best recall (1.0) and F1 (0.513) with lowest latency (37.4s)"
fig.text(0.5, 0.02, performance_text, ha='center', fontsize=10, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save
os.makedirs('results', exist_ok=True)
fig.savefig('results/combined_metrics_grouped.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('✓ Combined Metrics Chart saved: results/combined_metrics_grouped.png')

# Print summary
print("\n" + "="*80)
print("COMBINED CLASSIFICATION PERFORMANCE METRICS")
print("="*80)
print(f"{'Model':<30} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
print("-"*80)
for key, model_name in zip(model_keys, models):
    acc = data[key]['accuracy']
    prec = data[key]['precision']
    rec = data[key]['recall']
    f1_score = data[key]['f1']
    print(f"{key:<30} {acc:>12.4f} {prec:>12.4f} {rec:>12.4f} {f1_score:>12.4f}")
print("="*80 + "\n")

plt.close()