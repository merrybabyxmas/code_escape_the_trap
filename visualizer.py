import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import pearsonr

# ==========================================
# 논문용 전역 스타일 설정 (ECCV/CVPR Standard)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 20,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
})

OUTPUT_DIR = "/home/dongwoo43/paper_escapethetrap/escapethetrap/outputs"
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis_plots")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def plot_lpips_timeline(csv_path, shot_boundaries, model_name, sid, save_path):
    """1. LPIPS Timeline (Anti-Uniform Continuity Bias)"""
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 4))
    plt.plot(df['frame_idx'], df['lpips_dist'], color='#1f77b4', linewidth=2, label='Perceptual Dist.')
    for i, b in enumerate(shot_boundaries):
        plt.axvline(x=b, color='#d62728', linestyle='--', alpha=0.6, label='Shot Boundary' if i == 0 else "")
    plt.xlabel("Frame Index")
    plt.ylabel("LPIPS Distance")
    plt.title(f"Temporal Discontinuity: {model_name} ({sid})")
    plt.legend(); plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_dsa_heatmap(matrix, model_name, sid, save_path):
    """2. DSA Heatmap (Diagonal Alignment Proof)"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(matrix), annot=True, fmt=".2f", cmap="mako_r", vmin=0, vmax=1, cbar_kws={'label': 'Softmax Prob.'})
    plt.xlabel("Target Text Prompts"); plt.ylabel("Generated Video Shots")
    plt.title(f"DSA Heatmap: {model_name}\n({sid})")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_pareto_frontier(df, save_path):
    """3. Double-Kill Pareto Frontier (The Core Analysis)"""
    model_avg = df.groupby(['Model', 'Track']).agg({'Subj_Consistency': 'mean', 'BG_Diversity': 'mean', 'DSA_Score': 'mean'}).reset_index()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=model_avg, x='BG_Diversity', y='Subj_Consistency', hue='Model', style='Track', size='DSA_Score', sizes=(100, 500), alpha=0.8, palette='viridis')
    rect = plt.Rectangle((0.25, 0.90), 0.1, 0.08, linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect); plt.text(0.25, 0.99, "Unoccupied Ideal Zone", color='red', weight='bold')
    plt.xlabel("Background Diversity (Dynamic)"); plt.ylabel("Subject Consistency (Identity)")
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_correlation_matrix(df, save_path):
    """4. Independence Proof: Metrics Correlation"""
    metrics = ['Subj_Consistency', 'BG_Diversity', 'DSA_Score', 'Cut_Sharpness']
    sns.heatmap(df[metrics].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Metrics Independence Proof"); plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_tau_ablation(res_path, save_path):
    """5. Tau Ablation: Ranking Consistency"""
    with open(res_path, 'r') as f: data = json.load(f)
    tau_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    for model, samples in data.items():
        for tau in tau_vals:
            scores = [np.mean(np.diag(np.exp(np.array(r['metrics']['dsa_matrix'])/tau)/np.sum(np.exp(np.array(r['metrics']['dsa_matrix'])/tau), axis=1, keepdims=True))) for r in samples.values()]
            results.append({'Model': model, 'Tau': tau, 'Avg_DSA': np.mean(scores)})
    sns.lineplot(data=pd.DataFrame(results), x='Tau', y='Avg_DSA', hue='Model', marker='o')
    plt.xscale('log'); plt.title("Ranking Consistency (tau Ablation)"); plt.tight_layout(); plt.savefig(save_path); plt.close()

def generate_failure_gallery(df, save_dir):
    """6. Failure Gallery: Identity Amnesia, Static Trap, Morphing"""
    gallery_dir = os.path.join(save_dir, "failure_gallery")
    os.makedirs(gallery_dir, exist_ok=True)
    worst_subj = df.sort_values('Subj_Consistency').head(5)
    worst_bg = df.sort_values('BG_Diversity').head(5)
    worst_cut = df.sort_values('Cut_Sharpness').head(5)
    with open(os.path.join(gallery_dir, "failure_analysis.txt"), "w") as f:
        f.write("=== Failure Modes for Appendix ===\n\n[Identity Amnesia]\n" + worst_subj[['sid', 'Model', 'Subj_Consistency']].to_string() +
                "\n\n[Static Trap]\n" + worst_bg[['sid', 'Model', 'BG_Diversity']].to_string() +
                "\n\n[Morphing/Lazy Cut]\n" + worst_cut[['sid', 'Model', 'Cut_Sharpness']].to_string())

def run_analysis():
    print("📊 Running Full Meta-Analysis...")
    res_path = os.path.join(OUTPUT_DIR, "per_sample_results_huge.json")
    if not os.path.exists(res_path): return
    with open(res_path, 'r') as f: data = json.load(f)
    rows = []
    for model, samples in data.items():
        for sid, res in samples.items():
            rows.append({'sid': sid, 'Model': model, 'Subj_Consistency': res['metrics']['subj'], 'BG_Diversity': res['metrics']['bg'], 
                         'DSA_Score': np.mean(res['metrics']['diag']), 'Cut_Sharpness': res['metrics']['cut'], 'Visibility': res['metrics'].get('visibility', 1.0),
                         'Track': res['track'], 'Sub_Category': res['sub_category']})
    df = pd.DataFrame(rows)
    plot_correlation_matrix(df, os.path.join(ANALYSIS_DIR, "metrics_correlation.pdf"))
    plot_tau_ablation(res_path, os.path.join(ANALYSIS_DIR, "tau_ablation.pdf"))
    plot_pareto_frontier(df, os.path.join(ANALYSIS_DIR, "pareto_frontier.pdf"))
    generate_failure_gallery(df, ANALYSIS_DIR)
    df.groupby(['Model', 'Track']).mean().to_csv(os.path.join(OUTPUT_DIR, "final_table_2.csv"))
    print(f"✅ Analysis Complete. Results in {ANALYSIS_DIR}")

if __name__ == "__main__":
    run_analysis()
