import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
PAPER_ROOT = "/home/dongwoo38/paper_escape_the_trap/Escaping-the-Static-Video-Trap"
# ---------------------

def generate_bar_chart(data, title, filename, invert_metrics=[]):
    models = list(data.keys())
    if not models: return
    original_metrics = list(data[models[0]].keys())
    
    # Create DataFrame
    rows = []
    for model in models:
        for metric in original_metrics:
            val = data[model][metric]
            if metric in invert_metrics:
                val = 1.0 - val
                metric_name = f"1 - {metric.replace('_', ' ')}"
            else:
                metric_name = metric.replace("_", " ")
            rows.append({"Model": model, "Metric": metric_name, "Score": val})
            
    df = pd.DataFrame(rows)
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x="Metric", y="Score", hue="Model", palette="bright", alpha=0.9)
    plt.ylim(0, 1.05)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel("Score (Higher is Better)", fontsize=12)
    plt.xlabel("Evaluation Metrics", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Models")
    plt.tight_layout()
    
    os.makedirs(os.path.join(PAPER_ROOT, "figures"), exist_ok=True)
    save_path = os.path.join(PAPER_ROOT, "figures", filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved bar chart to {save_path}")
    plt.close()

def generate_rich_scatter_plots(data_a, data_b, filename):
    df_a = pd.DataFrame.from_dict(data_a, orient='index').reset_index().rename(columns={'index': 'Model'})
    df_b = pd.DataFrame.from_dict(data_b, orient='index').reset_index().rename(columns={'index': 'Model'})
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    sns.set_style("whitegrid")
    
    # --- Track S (Semantic Leap) ---
    # Plot 1: Div vs Cons
    size_a = df_a['Diagonal_Alignment'] * 800 + 100
    sns.scatterplot(data=df_a, x='Background_Diversity', y='Subject_Consistency', size=size_a, hue='Model', 
                    ax=axes[0, 0], sizes=(100, 1000), alpha=0.8, palette='bright', legend=False)
    axes[0, 0].set_title("Track S: Consistency vs Diversity", fontsize=16, fontweight='bold')
    axes[0, 0].set_xlabel(r"Background Diversity $\rightarrow$", fontsize=14)
    axes[0, 0].set_ylabel(r"Subject Consistency $\uparrow$", fontsize=14)
    axes[0, 0].set_xlim(-0.05, 1.05)
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].text(0.9, 0.95, "IDEAL GOAL", color='green', fontweight='bold', fontsize=12, ha='center')
    axes[0, 0].text(0.15, 0.9, "STATIC TRAP", color='red', fontweight='bold', fontsize=12, ha='center', alpha=0.6)
    for i in range(df_a.shape[0]): axes[0, 0].text(df_a.Background_Diversity.iloc[i]+0.02, df_a.Subject_Consistency.iloc[i], df_a.Model.iloc[i], fontsize=9)

    # Plot 2: DSA vs Cons
    size_a_2 = df_a['Background_Diversity'] * 800 + 100
    sns.scatterplot(data=df_a, x='Diagonal_Alignment', y='Subject_Consistency', size=size_a_2, hue='Model', 
                    ax=axes[0, 1], sizes=(100, 1000), alpha=0.8, palette='bright', legend=False)
    axes[0, 1].set_title("Track S: Consistency vs DSA", fontsize=16, fontweight='bold')
    axes[0, 1].set_xlabel(r"Diagonal Semantic Alignment (DSA) $\rightarrow$", fontsize=14)
    axes[0, 1].set_ylabel(r"Subject Consistency $\uparrow$", fontsize=14)
    axes[0, 1].set_xlim(-0.05, 1.05)
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].text(0.9, 0.95, "IDEAL DECOUPLED", color='green', fontweight='bold', fontsize=12, ha='center')
    axes[0, 1].text(0.1, 0.9, "STATIC TRAP", color='red', fontweight='bold', fontsize=12, ha='center', alpha=0.6)
    for i in range(df_a.shape[0]): axes[0, 1].text(df_a.Diagonal_Alignment.iloc[i]+0.02, df_a.Subject_Consistency.iloc[i], df_a.Model.iloc[i], fontsize=9)

    # --- Track M (Motion Continuity) ---
    # Plot 3: Div vs Cons
    size_b = df_b['Diagonal_Alignment'] * 800 + 100
    sns.scatterplot(data=df_b, x='Background_Diversity', y='Subject_Consistency', size=size_b, hue='Model', 
                    ax=axes[1, 0], sizes=(100, 1000), alpha=0.8, palette='bright', legend=False)
    axes[1, 0].set_title("Track M: Consistency vs Diversity", fontsize=16, fontweight='bold')
    axes[1, 0].set_xlabel(r"Background Diversity $\rightarrow$ (Lower is Better)", fontsize=14)
    axes[1, 0].set_ylabel(r"Subject Consistency $\uparrow$", fontsize=14)
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].text(0.1, 0.95, "IDEAL GOAL", color='green', fontweight='bold', fontsize=12, ha='center')
    axes[1, 0].text(0.85, 0.15, "HALLUCINATION", color='orange', fontweight='bold', fontsize=12, ha='center', alpha=0.6)
    for i in range(df_b.shape[0]): axes[1, 0].text(df_b.Background_Diversity.iloc[i]+0.02, df_b.Subject_Consistency.iloc[i], df_b.Model.iloc[i], fontsize=9)

    # Plot 4: DSA vs Cons
    size_b_2 = (1 - df_b['Background_Diversity']) * 800 + 100
    sns.scatterplot(data=df_b, x='Diagonal_Alignment', y='Subject_Consistency', size=size_b_2, hue='Model', 
                    ax=axes[1, 1], sizes=(100, 1000), alpha=0.8, palette='bright')
    axes[1, 1].set_title("Track M: Consistency vs DSA", fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel(r"Diagonal Semantic Alignment (DSA) $\rightarrow$", fontsize=14)
    axes[1, 1].set_ylabel(r"Subject Consistency $\uparrow$", fontsize=14)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].text(0.9, 0.95, "IDEAL DECOUPLED", color='green', fontweight='bold', fontsize=12, ha='center')
    axes[1, 1].text(0.1, 0.9, "STATIC TRAP", color='red', fontweight='bold', fontsize=12, ha='center', alpha=0.6)
    for i in range(df_b.shape[0]): axes[1, 1].text(df_b.Diagonal_Alignment.iloc[i]+0.02, df_b.Subject_Consistency.iloc[i], df_b.Model.iloc[i], fontsize=9)

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Models & Bubble Size")
    
    os.makedirs(os.path.join(PAPER_ROOT, "figures"), exist_ok=True)
    save_path = os.path.join(PAPER_ROOT, "figures", filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()

def main():
    path_a = os.path.join(OUTPUT_DIR, "final_metrics_results_set_a.json")
    path_b = os.path.join(OUTPUT_DIR, "final_metrics_results_set_b.json")
    if os.path.exists(path_a) and os.path.exists(path_b):
        with open(path_a, 'r') as f: data_a = json.load(f)
        with open(path_b, 'r') as f: data_b = json.load(f)
        generate_bar_chart(data_a, "Track S: Semantic Leap Performance", "bar_chart_set_a.pdf")
        generate_bar_chart(data_b, "Track M: Motion Continuity Performance", "bar_chart_set_b.pdf", invert_metrics=["Background_Diversity", "Cut_Sharpness"])
        generate_rich_scatter_plots(data_a, data_b, "rich_tradeoff_scatter.pdf")
    else:
        print(f"Error: JSON paths not found: {path_a} or {path_b}")
    print("✅ Bar charts and Rich 2x2 scatter plots generated successfully.")

if __name__ == "__main__":
    main()
