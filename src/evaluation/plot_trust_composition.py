"""
Trust Score Weight Distribution Graphic
Visualizes the weighted components that make up the trust score calculation.
"""

import matplotlib.pyplot as plt

# Trust score composition with abbreviated labels
components = [
    ('Relevance (Rq)', 40),
    ('Source Reliability (Rs)', 30),
    ('Cross-Document Consistency (C)', 30)
]

# Muted academic colors: light grey, dark grey, white
colors = ['#A9A9A9', '#696969', '#D3D3D3']  # Dark grey, Medium grey, Light grey

# Create figure with larger size
fig = plt.figure(figsize=(14, 6), dpi=150)
ax = fig.add_subplot(111)

# Create stacked horizontal bar
left = 0
bar_height = 0.8
y_pos = 0.8

bar_positions = []  # Store positions for labels

for i, (component_name, percentage) in enumerate(components):
    # Draw segment
    rect = ax.barh(y_pos, percentage, left=left, height=bar_height, 
                   color=colors[i], edgecolor='black', linewidth=2)
    
    # Store center position
    center_x = left + percentage / 2
    bar_positions.append((center_x, component_name, percentage, colors[i]))
    
    left += percentage

# Add centered percentage labels ON the bar
for center_x, component_name, percentage, color in bar_positions:
    text_color = 'white' if color != '#D3D3D3' else 'black'
    ax.text(center_x, y_pos + 0.05, f'{percentage}%', 
            ha='center', va='center', fontsize=22, fontweight='bold', 
            color=text_color)

# Add component names below with clear spacing
for center_x, component_name, percentage, color in bar_positions:
    ax.text(center_x, y_pos - 0.55, component_name, 
            ha='center', va='top', fontsize=13, fontweight='bold', 
            color='#333333')

# Add trust score formula below
formula_y = y_pos - 1.4
ax.text(0.5, formula_y, r'$T = 0.4R_q + 0.3R_s + 0.3C$', 
        ha='center', va='top', fontsize=18, fontweight='bold',
        color='#333333', 
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', 
                  edgecolor='#999999', linewidth=1.5))

# Set axis properties
ax.set_xlim(-5, 105)
ax.set_ylim(-1.8, 1.5)
ax.axis('off')

# Title
fig.suptitle('Trust Score Weight Distribution', fontsize=20, fontweight='bold', y=0.98)

# Tight layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
output_path = 'results/trust_composition.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Trust score weight distribution graphic saved: {output_path}")

# Summary
print("\n" + "="*60)
print("TRUST SCORE WEIGHT DISTRIBUTION")
print("="*60)
print("Formula: T = 0.4Rq + 0.3Rs + 0.3C")
print("="*60)
for component_name, percentage in components:
    print(f"{component_name:.<40} {percentage:>3}%")
print("-"*60)
print(f"{'Total':.<40} {sum([p for _, p in components]):>3}%")
print("="*60)

plt.close()
