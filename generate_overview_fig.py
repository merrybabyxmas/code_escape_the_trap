import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

PAPER_FIGURE_PATH = "/home/dongwoo38/paper_escape_the_trap/Escaping-the-Static-Video-Trap/figures/fig1_overview.pdf"

def create_overview_figure():
    # Set up the figure
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid layout: 3 main sections
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.3)
    
    # ---------------------------------------------------------
    # TOP SECTION: A. Failure Modes (Problem Exposure)
    # ---------------------------------------------------------
    ax_top = fig.add_subplot(gs[0])
    ax_top.axis('off')
    
    # Section Title
    ax_top.text(-0.05, 1.05, "A. Failure Modes in Multi-Shot Video Generation (e.g., Track S: Semantic Leap)", 
                fontsize=16, fontweight='bold', ha='left', va='center')
    
    # Input Prompts Box
    prompt_box = patches.FancyBboxPatch((-0.05, 0.2), 0.35, 0.7, boxstyle="round,pad=0.02", 
                                        ec="black", fc="#E8F4F8", lw=1.5)
    ax_top.add_patch(prompt_box)
    ax_top.text(0.125, 0.8, "Input Narrative Prompts", fontsize=14, fontweight='bold', ha='center')
    
    p1 = "Shot 1:\n\"A cybernetic samurai in a\ntraditional Japanese garden...\""
    p2 = "Shot 2:\n\"A cybernetic samurai in a\nneon Cyberpunk city...\""
    
    ax_top.text(0.125, 0.55, p1, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    ax_top.text(0.125, 0.35, p2, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Arrow
    ax_top.arrow(0.33, 0.55, 0.05, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    # Create fake images for illustration (user will replace with actual)
    img_garden = np.ones((100, 150, 3)) * [0.6, 0.8, 0.6] # Greenish (Garden)
    img_city = np.ones((100, 150, 3)) * [0.3, 0.2, 0.5] # Purplish (City)
    img_morphed = np.ones((100, 150, 3)) * [0.5, 0.5, 0.5] # Gray (Morphed)
    
    # 1. Static Trap
    ax_top.text(0.5, 0.9, "Failure 1: Static Trap", fontsize=12, fontweight='bold', ha='center')
    ax_top.imshow(img_garden, extent=[0.4, 0.48, 0.4, 0.8]) # Shot 1
    ax_top.imshow(img_garden, extent=[0.52, 0.6, 0.4, 0.8]) # Shot 2 (Failed to change)
    ax_top.text(0.5, 0.3, "High Continuity Score (0.99)\nBut ignores prompt 2", fontsize=10, ha='center', color='red')
    ax_top.arrow(0.48, 0.6, 0.04, 0, head_width=0.03, head_length=0.01, fc='black', ec='black')
    
    # 2. Identity Amnesia
    ax_top.text(0.8, 0.9, "Failure 2: Identity Amnesia", fontsize=12, fontweight='bold', ha='center')
    ax_top.imshow(img_garden, extent=[0.7, 0.78, 0.4, 0.8]) # Shot 1
    ax_top.imshow(img_city, extent=[0.82, 0.9, 0.4, 0.8]) # Shot 2 (Changed but character morphed)
    # Add a red marker for character change
    ax_top.plot([0.86], [0.6], 'rx', markersize=15, markeredgewidth=3)
    ax_top.text(0.8, 0.3, "High Diversity Score\nBut character is lost", fontsize=10, ha='center', color='orange')
    ax_top.arrow(0.78, 0.6, 0.04, 0, head_width=0.03, head_length=0.01, fc='black', ec='black')
    
    # ---------------------------------------------------------
    # CENTER SECTION: B. Proposed Pipeline (DSA)
    # ---------------------------------------------------------
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.axis('off')
    ax_mid.text(-0.05, 1.05, "B. The DIAL Pipeline & Diagonal Semantic Alignment (DSA)", 
                fontsize=16, fontweight='bold', ha='left', va='center')
    
    # Base Box
    pipe_box = patches.FancyBboxPatch((-0.05, 0.1), 1.0, 0.8, boxstyle="round,pad=0.02", 
                                      ec="black", fc="#F0F8EA", lw=1.5)
    ax_mid.add_patch(pipe_box)
    
    # Matrix M
    ax_mid.text(0.15, 0.75, "Similarity Matrix $M$", fontsize=12, fontweight='bold', ha='center')
    ax_mid.text(0.15, 0.5, "[ 0.85   0.40 ]\n[ 0.30   0.75 ]", fontsize=20, ha='center', va='center', family='monospace')
    ax_mid.text(0.15, 0.25, "Absolute Scores\n(Vulnerable to Bleeding)", fontsize=10, ha='center')
    
    # Operator
    ax_mid.arrow(0.3, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')
    ax_mid.text(0.5, 0.5, r"$P_{i,j} = \frac{\exp(\tau \cdot M_{i,j})}{\sum_{k=1}^{K} \exp(\tau \cdot M_{k,j})}$", 
                fontsize=18, ha='center', va='center', bbox=dict(facecolor='white', boxstyle='round,pad=0.5'))
    ax_mid.text(0.5, 0.75, "Column-wise Softmax\n(Contrastive Isolation)", fontsize=12, fontweight='bold', ha='center')
    
    # Heatmap P
    ax_mid.arrow(0.7, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')
    ax_mid.text(0.85, 0.75, "DSA Heatmap $P$", fontsize=12, fontweight='bold', ha='center')
    
    # Draw a stylized heatmap
    h_data = np.array([[0.95, 0.05], [0.05, 0.95]])
    ax_mid.imshow(h_data, cmap='Blues', extent=[0.75, 0.95, 0.3, 0.6], vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            val = h_data[1-i, j]
            color = "white" if val > 0.5 else "black"
            ax_mid.text(0.75 + 0.1*j + 0.05, 0.3 + 0.15*i + 0.075, f"{val:.2f}", color=color, ha='center', va='center', fontweight='bold')
    
    ax_mid.text(0.85, 0.15, r"Diagonal enforces zero-sum" + "\n" + r"independence ($DSA \rightarrow 1.0$)", fontsize=10, ha='center', color='green')

    # ---------------------------------------------------------
    # BOTTOM SECTION: C. Comparative Analysis
    # ---------------------------------------------------------
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.axis('off')
    ax_bot.text(-0.05, 1.05, "C. Legacy vs. DIAL Metrics: Unveiling the \"Double-Kill\" Dilemma", 
                fontsize=16, fontweight='bold', ha='left', va='center')
    
    # Legacy Metrics (Bar Chart mock)
    ax_bot.text(0.2, 0.9, "Legacy Metrics (Holistic)", fontsize=14, fontweight='bold', ha='center')
    ax_bot.bar([0.1, 0.3], [0.95, 0.92], width=0.1, color=['#ff9999', '#99ccff'])
    ax_bot.text(0.1, 0.45, "FVD\n(Static Model)", fontsize=10, ha='center', va='top')
    ax_bot.text(0.3, 0.45, "CLIP\n(Amnesia Model)", fontsize=10, ha='center', va='top')
    ax_bot.text(0.2, 0.1, "Masks failures by\nrewarding uniform continuity", fontsize=11, ha='center', style='italic')

    ax_bot.arrow(0.45, 0.5, 0.05, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    # DIAL Pareto Frontier
    ax_bot.text(0.75, 0.9, "DIAL Pareto Frontier (Track S)", fontsize=14, fontweight='bold', ha='center')
    
    # Draw mini scatter plot
    ax_scatter = fig.add_axes([0.55, 0.1, 0.4, 0.7])
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_xlabel("Background Diversity", fontsize=10)
    ax_scatter.set_ylabel("Subject Consistency", fontsize=10)
    
    # Static Trap cluster
    ax_scatter.scatter([0.1, 0.15, 0.05], [0.9, 0.85, 0.95], color='red', s=100, alpha=0.7, label='Foundation Models')
    ax_scatter.text(0.1, 0.75, "Static\nTrap", color='red', ha='center', fontweight='bold')
    
    # Identity Amnesia cluster
    ax_scatter.scatter([0.8, 0.75, 0.9], [0.2, 0.15, 0.3], color='orange', s=100, alpha=0.7, label='T2MSV Frameworks')
    ax_scatter.text(0.8, 0.4, "Identity\nAmnesia", color='orange', ha='center', fontweight='bold')
    
    # Ideal Target
    ax_scatter.scatter([0.95], [0.95], color='green', marker='*', s=300, label='Ideal Goal')
    ax_scatter.text(0.8, 0.9, "Unoccupied\nIdeal Zone", color='green', ha='center', fontweight='bold')
    
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure
    os.makedirs(os.path.dirname(PAPER_FIGURE_PATH), exist_ok=True)
    plt.savefig(PAPER_FIGURE_PATH, bbox_inches='tight', dpi=300)
    print(f"✅ Generated high-quality overview figure at: {PAPER_FIGURE_PATH}")

if __name__ == "__main__":
    create_overview_figure()
