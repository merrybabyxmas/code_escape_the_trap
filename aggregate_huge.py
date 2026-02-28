import json
import os
import numpy as np

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
PER_SAMPLE_PATH = os.path.join(OUTPUT_DIR, "per_sample_results_huge.json")
# ---------------------

def main():
    if not os.path.exists(PER_SAMPLE_PATH):
        print(f"Error: {PER_SAMPLE_PATH} not found.")
        return

    with open(PER_SAMPLE_PATH, 'r') as f:
        data = json.load(f)

    summary_a = {}
    summary_b = {}

    for model_name, scenarios in data.items():
        metrics_a = {"Subject_Consistency": [], "Background_Diversity": [], "Diagonal_Alignment": [], "Cut_Sharpness": []}
        metrics_b = {"Subject_Consistency": [], "Background_Diversity": [], "Diagonal_Alignment": [], "Cut_Sharpness": []}

        for sid, entry in scenarios.items():
            m = entry['metrics']
            # Using track_s and track_m based on sample IDs
            if "track_s" in sid:
                target = metrics_a
            elif "track_m" in sid:
                target = metrics_b
            else:
                continue # Skip if neither
            
            target["Subject_Consistency"].append(m['subj'])
            target["Background_Diversity"].append(m['bg'])
            target["Diagonal_Alignment"].append(m['diag'])
            target["Cut_Sharpness"].append(m['cut'])

        # Aggregate averages
        if metrics_a["Subject_Consistency"]:
            summary_a[model_name] = {k: float(np.mean(v)) for k, v in metrics_a.items()}
        if metrics_b["Subject_Consistency"]:
            summary_b[model_name] = {k: float(np.mean(v)) for k, v in metrics_b.items()}

    # Save finalized summaries
    with open(os.path.join(OUTPUT_DIR, "final_metrics_results_set_a.json"), "w") as f:
        json.dump(summary_a, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "final_metrics_results_set_b.json"), "w") as f:
        json.dump(summary_b, f, indent=4)

    print(f"✅ Finalized aggregation for {len(summary_a)} models in Set A (Track S) and {len(summary_b)} models in Set B (Track M).")

if __name__ == "__main__":
    main()
