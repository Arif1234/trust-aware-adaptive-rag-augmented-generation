import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig = plt.figure(figsize=(13, 16), dpi=150, facecolor='white', edgecolor='none')
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

header_color = '#2C3E50'
header_bg = '#ECF0F1'
section_bg = '#F8F9FA'
border_color = '#34495E'
text_color = '#2C3E50'

y = 19.5

ax.text(5, y, 'Figure 4.5: Evidence-Grounded Prompt Construction', 
        fontsize=16, fontweight='bold', ha='center', color=header_color)
y -= 1.0

# SECTION 1: QUERY
rect = FancyBboxPatch((0.3, y-0.35), 9.4, 0.45, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=header_bg, linewidth=2)
ax.add_patch(rect)
ax.text(0.5, y-0.15, '1. USER QUERY', fontsize=12, fontweight='bold', color=header_color, va='center')
y -= 0.7

rect = FancyBboxPatch((0.3, y-1.0), 9.4, 0.8, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=section_bg, linewidth=1.5)
ax.add_patch(rect)
ax.text(0.6, y-0.25, '"What are the benefits of Retrieval-Augmented Generation?"', fontsize=10, family='monospace', color=text_color, va='center')
y -= 1.2

# SECTION 2: PROMPT INSTRUCTIONS
rect = FancyBboxPatch((0.3, y-0.35), 9.4, 0.45, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=header_bg, linewidth=2)
ax.add_patch(rect)
ax.text(0.5, y-0.15, '2. PROMPT INSTRUCTIONS', fontsize=12, fontweight='bold', color=header_color, va='center')
y -= 0.7

rect = FancyBboxPatch((0.3, y-1.3), 9.4, 1.1, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=section_bg, linewidth=1.5)
ax.add_patch(rect)

instructions = ['• Answer ONLY using the provided context.', '• Do not generate unsupported information.', '• If the answer is unavailable, state that explicitly.']
instr_y = y - 0.3
for line in instructions:
    ax.text(0.6, instr_y, line, fontsize=9, family='monospace', color=text_color, va='top')
    instr_y -= 0.32
y -= 1.5

# SECTION 3: RETRIEVED CONTEXT
rect = FancyBboxPatch((0.3, y-0.35), 9.4, 0.45, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=header_bg, linewidth=2)
ax.add_patch(rect)
ax.text(0.5, y-0.15, '3. RETRIEVED AND ACCEPTED EVIDENCE', fontsize=12, fontweight='bold', color=header_color, va='center')
y -= 0.7

rect = FancyBboxPatch((0.3, y-5.2), 9.4, 5.0, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=section_bg, linewidth=1.5)
ax.add_patch(rect)

context_text = ['[Source: 01_RAG_Paper.pdf - Lewis et al., 2020]', 'RAG retrieves knowledge from external information sources to augment', 'the generation process. Retrieved documents are utilized as context', 'for the language generation model. The retriever is a dense passage', 'retriever (DPR), and the generator is BART.', '', '[Source: 03_Self_RAG.pdf - Asai et al., 2024]', 'Self-RAG introduces adaptive retrieval, activating knowledge retrieval', 'on an as-needed basis. This enables more efficient knowledge augmentation']

ctx_y = y - 0.3
for line in context_text:
    if ctx_y > y - 4.8:
        if line.startswith('[Source'):
            ax.text(0.6, ctx_y, line, fontsize=8, fontweight='bold', family='monospace', color='#5D6D7B', va='top')
        elif line:
            ax.text(0.6, ctx_y, line, fontsize=8, family='monospace', color=text_color, va='top')
        ctx_y -= 0.28
y -= 5.5

# SECTION 4: GENERATED ANSWER
rect = FancyBboxPatch((0.3, y-0.35), 9.4, 0.45, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=header_bg, linewidth=2)
ax.add_patch(rect)
ax.text(0.5, y-0.15, '4. GENERATED ANSWER', fontsize=12, fontweight='bold', color=header_color, va='center')
y -= 0.7

rect = FancyBboxPatch((0.3, y-3.8), 9.4, 3.6, boxstyle='round,pad=0.05',
                       edgecolor=border_color, facecolor=section_bg, linewidth=1.5)
ax.add_patch(rect)

answer_text = ['The key benefits of Retrieval-Augmented Generation:', '', '1. Factual Grounding: Reduces hallucinations by augmenting generation', '   with retrieved external knowledge.', '', '2. Hybrid Knowledge: Combines learned model weights (parametric) with', '   external documents (non-parametric knowledge).', '', '3. Adaptive Retrieval: Self-RAG enables on-demand retrieval,', '   improving efficiency.']

ans_y = y - 0.3
for line in answer_text:
    if ans_y > y - 3.5:
        if line:
            ax.text(0.6, ans_y, line, fontsize=8.5, family='monospace', color=text_color, va='top')
        ans_y -= 0.28

plt.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/evidence_grounded_prompt_figure.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('✓ Figure 4.5 saved: results/evidence_grounded_prompt_figure.png')
plt.close()
