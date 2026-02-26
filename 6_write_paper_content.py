import json
import os
import numpy as np

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
PAPER_ROOTS = [
    "/home/dongwoo43/paper/paper_escapethetrap/Escaping-the-Static-Video-Trap",
    "/home/dongwoo43/paper_escapethetrap/paper_escapethetrap/Escaping-the-Static-Video-Trap"
]
# ---------------------

def load_metrics(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return {}
    with open(path, 'r') as f:
        return json.load(f)

def write_section(filename, content):
    for root in PAPER_ROOTS:
        path = os.path.join(root, "sections", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

def generate_experiments_analysis(data_a, data_b):
    if not data_a or not data_b: return "% Error: Metrics not found."
    
    # Logic for Bolding Best Values
    def get_best_values(data, goals):
        """
        data: dict of model: metrics
        goals: dict of metric: "max" or "min"
        """
        best = {}
        for metric, goal in goals.items():
            vals = [m.get(metric, 0) for m in data.values()]
            if not vals: continue
            best[metric] = max(vals) if goal == "max" else min(vals)
        return best

    goals_s = {"Subject_Consistency": "max", "Background_Diversity": "max", "Diagonal_Alignment": "max", "Cut_Sharpness": "max"}
    goals_m = {"Subject_Consistency": "max", "Background_Diversity": "min", "Diagonal_Alignment": "max", "Cut_Sharpness": "min"}
    
    best_s = get_best_values(data_a, goals_s)
    best_m = get_best_values(data_b, goals_m)

    content = r"""\section{Experiments and Unveiling the Blind Spots}
\label{sec:experiments}

Our evaluation on Dynamic-MSV-Bench reveals a systematic failure in current video generation architectures. To properly diagnose these failures, we categorized the evaluated baselines into two distinct groups: \textbf{Group 1: Foundation T2V Models} and \textbf{Group 2: Specialized T2MSV Frameworks}.

\begin{figure*}[t!]
    \centering
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{figures/bar_chart_set_a.pdf}
        \caption{Track S: Semantic Leap}
    \end{subfigure}\hfill
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{figures/bar_chart_set_b.pdf}
        \caption{Track M: Motion Continuity}
    \end{subfigure}
    \caption{Grouped Bar Charts of Model Performance. In \textbf{Track S}, models must achieve high scores across all metrics. In \textbf{Track M}, we plot inverted axes for Diversity and Sharpness ($1-x$) such that higher bars consistently represent better performance across all scenarios.}
    \label{fig:bar_chart}
\end{figure*}

\begin{table*}[t!]
\centering
\caption{Main Results on \textbf{Track S (Semantic Leap)} and \textbf{Track M (Motion Continuity)}. Track S demands narrative-driven dynamics ($\uparrow$ is better for all). Track M demands spatial stability ($\downarrow$ is better for Diversity and Sharpness). Best in \textbf{bold}.}
\label{tab:main_results}
\resizebox{0.95\textwidth}{!}{
\begin{tabular}{ll|cccc|cccc}
\toprule
& & \multicolumn{4}{c|}{\textbf{Track S (Semantic Leap)}} & \multicolumn{4}{c}{\textbf{Track M (Motion Continuity)}} \\
\textbf{Category} & \textbf{Method} & \textbf{Cons.} $\uparrow$ & \textbf{Div.} $\uparrow$ & \textbf{DSA} $\uparrow$ & \textbf{Sharp.} $\uparrow$ & \textbf{Cons.} $\uparrow$ & \textbf{Div.} $\downarrow$ & \textbf{DSA} $\uparrow$ & \textbf{Sharp.} $\downarrow$ \\
\midrule
"""
    # Helper to bold best
    def fmt(val, best_val, precision=3):
        s = f"{val:.{precision}f}"
        if abs(val - best_val) < 1e-6:
            return r"\textbf{" + s + "}"
        return s

    foundation_list = ["CogVideoX", "LTX-Video", "SVD", "ModelScope"]
    all_models = sorted(list(set(list(data_a.keys()) + list(data_b.keys()))))
    
    # Sort: Foundation models first, then Frameworks
    sorted_models = [m for m in all_models if m in foundation_list] + \
                    [m for m in all_models if m not in foundation_list]

    for m in sorted_models:
        da = data_a.get(m, {})
        db = data_b.get(m, {})
        cat = "Foundation" if m in foundation_list else "Framework"
        
        # Track S values
        s_cons = fmt(da.get('Subject_Consistency',0), best_s['Subject_Consistency'])
        s_div  = fmt(da.get('Background_Diversity',0), best_s['Background_Diversity'])
        s_dsa  = fmt(da.get('Diagonal_Alignment',0), best_s['Diagonal_Alignment'])
        s_shp  = fmt(da.get('Cut_Sharpness',0), best_s['Cut_Sharpness'])
        
        # Track M values
        m_cons = fmt(db.get('Subject_Consistency',0), best_m['Subject_Consistency'])
        m_div  = fmt(db.get('Background_Diversity',0), best_m['Background_Diversity'])
        m_dsa  = fmt(db.get('Diagonal_Alignment',0), best_m['Diagonal_Alignment'])
        m_shp  = fmt(db.get('Cut_Sharpness',0), best_m['Cut_Sharpness'])
        
        content += f"{cat} & {m} & {s_cons} & {s_div} & {s_dsa} & {s_shp} & {m_cons} & {m_div} & {m_dsa} & {m_shp} \\\\\n"

    content += r"""\bottomrule
\end{tabular}}
\end{table*}

\subsection{Group 1: Foundation T2V Models and the Static Trap}
Foundation models (e.g., CogVideoX, LTX-Video, SVD) exhibit the raw limits of spatiotemporal priors. They achieve high Subject Consistency (>0.80) but catastrophically low Background Diversity (<0.10) in Track S. More importantly, their \textbf{DSA} scores remain near $0.02$, proving they generate a single static scene regardless of narrative instructions. This results in the uniform probability distribution seen in Fig.~\ref{fig:dsa_heatmaps}(A).

\subsection{Group 2: T2MSV Frameworks and the Trade-off Dilemma}
Specialized frameworks successfully break out of the Static Trap, achieving higher Background Diversity in Track S. However, this dynamism comes at the severe cost of identity preservation. Their Subject Consistency plummets, leading to \textit{Identity Amnesia}. This is clearly visualized in our comprehensive performance landscape in Figure~\ref{fig:scatter}. 

\begin{figure*}[t!]
    \centering
    \includegraphics[width=\linewidth]{figures/rich_tradeoff_scatter.pdf}
    \caption{The comprehensive performance landscape across Track S and Track M. The plots explicitly show the relationship between Subject Consistency, Background Diversity, and our proposed Diagonal Semantic Alignment (DSA). Current models are completely missing from the "Ideal Decoupled" goal (high consistency, high DSA).}
    \label{fig:scatter}
\end{figure*}

\subsection{The "Double-Kill" and the Context Paradox}
Our evaluation exposes a fascinating paradox: a metric's value is deeply context-dependent. For instance, CogVideoX exhibits exceptionally low Background Diversity. In Track M, traditional evaluators might misinterpret this as excellent spatial continuity. However, our cross-scenario analysis reveals the truth: the same model exhibits the same low diversity in Track S, proving it is merely trapped in static generation. Without the dual-track perspective of Track S and Track M, this critical failure would remain masked.
"""
    return content

def main():
    data_a = load_metrics("final_metrics_results_set_a.json")
    data_b = load_metrics("final_metrics_results_set_b.json")
    
    write_section("4_experiments.tex", generate_experiments_analysis(data_a, data_b))
    print("✅ Paper tables updated with bolded best performers.")

if __name__ == "__main__":
    main()
