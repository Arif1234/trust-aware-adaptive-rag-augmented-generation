import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Load model comparison data
with open("results/model_comparison.json", "r") as f:
    data = json.load(f)

# Extract F1 scores for waterfall
f1_baseline = data['Model_A_Baseline']['f1']           # 0.000
f1_adaptive = data['Model_B_Adaptive']['f1']           # 0.000
f1_trust = data['Model_C_Adaptive_Trust']['f1']        # 0.452
f1_proposed = data['Model_D_Proposed']['f1']           # 0.513

# Calculate contributions
stages = ['Baseline\nF1', 'Adaptive\nRetrieval', 'Trust\nScoring', 'Verification', 'Final\nProposed']
values = [f1_baseline, 0, f1_trust - f1_adaptive, f1_proposed - f1_trust, 0]
cumulative = [f1_baseline, f1_adaptive, f1_trust, f1_proposed, f1_proposed]

# Colors: grey for baseline, green for improvements, blue for final
colors = ['#95a5a6', '#2ecc71', '#2ecc71', '#2ecc71', '#3498db']

# Create figure
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

# Draw waterfall bars
bar_width = 0.6
x_pos = np.arange(len(stages))

for i, (stage, value, cum, color) in enumerate(zip(stages, values, cumulative, colors)):
    if i == 0:
        # Starting bar (baseline)
        ax.bar(x_pos[i], cum, bar_width, label='Starting Point', 
               color=color, edgecolor='black', linewidth=2, zorder=3)
        ax.text(x_pos[i], cum + 0.03, f'{cum:.3f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    elif i == len(stages) - 1:
        # Final bar
        ax.bar(x_pos[i], cum, bar_width, label='Final Result',
               color=color, edgecolor='black', linewidth=2, zorder=3)
        ax.text(x_pos[i], cum + 0.03, f'{cum:.3f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    else:
        # Contribution bars
        prev_cum = cumulative[i-1]
        if value > 0:
            ax.bar(x_pos[i], value, bar_width, bottom=prev_cum,
                   color=color, edgecolor='black', linewidth=1.5, zorder=3)
            ax.text(x_pos[i], prev_cum + value/2, f'+{value:.3f}', 
                   ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        else:
            # No improvement
            ax.bar(x_pos[i], 0.01, bar_width, bottom=prev_cum,
                   color='#e74c3c', edgecolor='black', linewidth=1.5, zorder=3, alpha=0.6)
            ax.text(x_pos[i], prev_cum - 0.04, 'No change', 
                   ha='center', va='top', fontsize=10, style='italic', color='#c0392b')
    
    # Connect bars with lines
    if i < len(stages) - 2:
        next_cum = cumulative[i+1]
        ax.plot([x_pos[i] + bar_width/2, x_pos[i+1] - bar_width/2],
               [cum, cum], 'k--', linewidth=1, alpha=0.5, zorder=1)

# Add milestone annotations
milestones = [
    (0, f1_baseline, f'Baseline\n{f1_baseline:.3f}'),
    (2, f1_trust, f'Trust Score\nAdded\n{f1_trust:.3f}'),
    (3, f1_proposed, f'Verification\nAdded\n{f1_proposed:.3f}')
]

for x, y, label in milestones:
    if y > 0:
        ax.annotate('', xy=(x, y), xytext=(x, y + 0.12),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#2c3e50'))

# Customize plot
ax.set_xlabel('Component', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Figure 5.x: Incremental F1 Improvement - Waterfall Contribution Analysis',
            fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=11, fontweight='bold')
ax.set_ylim(-0.1, 0.65)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add performance summary table
summary_text = (
    "Model A (Baseline):         F1 = 0.000  (Uniform retrieval, no verification)\n"
    "Model B (Adaptive):          F1 = 0.000  (Adaptive retrieval, no trust/verification)\n"
    "Model C (Adaptive+Trust):     F1 = 0.452  (Trust scoring enables filtering)\n"
    "Model D (Proposed):          F1 = 0.513  (Verification adds claim validation)"
)

props = dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9, pad=1)
ax.text(0.98, 0.35, summary_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='top', horizontalalignment='right',
       bbox=props, family='monospace', linespacing=1.8)

# Add key insights
insight_text = (
    "✓ Trust scoring alone improves F1 by 0.452 (enabling selective generation)\n"
    "✓ Verification refinement adds 0.061 more (achieving 1.0 recall with higher precision)"
)
ax.text(0.02, 0.05, insight_text, transform=ax.transAxes,
       fontsize=9, style='italic', color='#27ae60', fontweight='bold')

plt.tight_layout()

# Save
os.makedirs('results', exist_ok=True)
fig.savefig('results/f1_waterfall_contribution.png', dpi=150, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print('✓ F1 Waterfall Chart saved: results/f1_waterfall_contribution.png')

# Print summary
print("\n" + "="*80)
print("F1 SCORE INCREMENTAL IMPROVEMENT")
print("="*80)
print(f"Baseline F1 (Model A):                    {f1_baseline:.4f}")
print(f"After Adaptive Retrieval (Model B):       {f1_adaptive:.4f}  (Δ = {f1_adaptive - f1_baseline:+.4f})")
print(f"After Trust Scoring (Model C):            {f1_trust:.4f}  (Δ = {f1_trust - f1_adaptive:+.4f})")
print(f"After Verification (Model D - Proposed):  {f1_proposed:.4f}  (Δ = {f1_proposed - f1_trust:+.4f})")
print("="*80)
print(f"Total Improvement: {f1_proposed - f1_baseline:.4f} (+{(f1_proposed - f1_baseline)*100:.1f}%)")
print("="*80 + "\n")

plt.close()
