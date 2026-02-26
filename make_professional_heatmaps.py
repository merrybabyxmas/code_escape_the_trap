import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
FIGURES_DIR = "/home/dongwoo43/paper_escapethetrap/paper_escapethetrap/Escaping-the-Static-Video-Trap/figures"
# ---------------------

def generate_professional_heatmaps():
    K = 4
    sns.set_theme(style="white")
    
    # Map A: Static Trap (Random/Uniform confusion)
    # The model can't distinguish, so similarity is roughly equal everywhere
    mat_a = np.random.uniform(0.23, 0.27, (K, K))
    # Normalize rows to sum to 1 (softmax simulation)
    mat_a = mat_a / mat_a.sum(axis=1, keepdims=True)
    
    # Map B: Amnesia / Prompt Bleeding
    # Diagonal is visible but off-diagonal is high
    mat_b = np.array([
        [0.45, 0.25, 0.15, 0.15],
        [0.20, 0.40, 0.25, 0.15],
        [0.15, 0.20, 0.40, 0.25],
        [0.10, 0.15, 0.25, 0.50]
    ])
    
    # Map C: Ideal Decoupled Model
    # Sharp diagonal, near-zero off-diagonal
    mat_c = np.eye(K) * 0.9 + 0.025
    mat_c = mat_c / mat_c.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    labels = [f"P{i+1}" for i in range(K)]
    shots = [f"S{i+1}" for i in range(K)]

    # Plot A
    sns.heatmap(mat_a, annot=True, fmt=".2f", cmap=cmap, cbar=False, ax=axes[0], xticklabels=labels, yticklabels=shots, vmin=0, vmax=1)
    axes[0].set_title("(A) Static Trap\n(Uniform Confusion)", fontsize=16, fontweight='bold')
    
    # Plot B
    sns.heatmap(mat_b, annot=True, fmt=".2f", cmap=cmap, cbar=False, ax=axes[1], xticklabels=labels, yticklabels=shots, vmin=0, vmax=1)
    axes[1].set_title("(B) Context Bleeding\n(Partial Amnesia)", fontsize=16, fontweight='bold')
    
    # Plot C
    sns.heatmap(mat_c, annot=True, fmt=".2f", cmap=cmap, cbar=True, ax=axes[2], xticklabels=labels, yticklabels=shots, vmin=0, vmax=1)
    axes[2].set_title("(C) Ideal Model\n(Decoupled Dynamics)", fontsize=16, fontweight='bold')

    for ax in axes:
        ax.set_xlabel("Prompts", fontsize=14)
        ax.set_ylabel("Video Shots", fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "dsa_heatmaps_phd.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Professional Heatmaps saved to {out_path}")

if __name__ == "__main__":
    generate_professional_heatmaps()
