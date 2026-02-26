import os
import json
import csv
import random
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
EVAL_DIR = os.path.join(SCRIPT_DIR, "human_evaluation")
PER_SAMPLE_RESULTS_PATH = os.path.join(OUTPUT_DIR, "per_sample_results_full.json")

def load_json(path, default={}):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try: return json.load(f)
            except: return default
    return default

def main():
    print("Setting up Human Evaluation environment...")
    os.makedirs(EVAL_DIR, exist_ok=True)
    video_pool_dir = os.path.join(EVAL_DIR, "video_samples")
    os.makedirs(video_pool_dir, exist_ok=True)
    
    per_sample = load_json(PER_SAMPLE_RESULTS_PATH)
    if not per_sample:
        print("No per-sample results found! Run evaluation first.")
        return

    # Gather all available videos
    all_videos = []
    for model_name, entries in per_sample.items():
        for sid, data in entries.items():
            vid_path = os.path.join(OUTPUT_DIR, model_name, f"{sid}.mp4")
            if os.path.exists(vid_path):
                track = "Track S (Leap)" if "set_a" in sid else "Track M (Motion)"
                prompt_text = " | ".join(data.get("prompts", []))
                all_videos.append({
                    "model": model_name,
                    "sid": sid,
                    "track": track,
                    "prompt": prompt_text,
                    "path": vid_path,
                    "dsa_score": data["metrics"].get("diag", 0.0)
                })

    if not all_videos:
        print("No mp4 videos found in output directories.")
        return

    # Randomly sample 30 videos for a robust human evaluation study
    sample_size = min(30, len(all_videos))
    sampled_videos = random.sample(all_videos, sample_size)
    
    csv_path = os.path.join(EVAL_DIR, "human_evaluation_sheet.csv")
    
    # Create CSV template
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Evaluation_ID", "Video_Filename", "Track", "Model_Name(Hidden)", "Prompt",
            "Human_Subject_Consistency(1-5)", 
            "Human_Background_Dynamics(1-5)", 
            "Human_Prompt_Adherence(1-5)", 
            "Human_Cut_Sharpness(1-5)", 
            "Human_Overall_Quality(1-5)",
            "Notes_or_Amnesia_Observed(O/X)"
        ])
        
        for idx, vid in enumerate(sampled_videos):
            eval_id = f"Eval_{idx+1:03d}"
            ext = os.path.splitext(vid["path"])[1]
            neutral_filename = f"{eval_id}{ext}"
            dest_path = os.path.join(video_pool_dir, neutral_filename)
            shutil.copy2(vid["path"], dest_path)
            
            writer.writerow([
                eval_id, neutral_filename, vid["track"], vid["model"], vid["prompt"],
                "", "", "", "", "", ""
            ])

    # Create a README with instructions for the evaluator
    readme_path = os.path.join(EVAL_DIR, "EVALUATION_INSTRUCTIONS.md")
    with open(readme_path, 'w') as f:
        f.write("# Human Evaluation Protocol\n\n")
        f.write("Please watch the videos in the `video_samples` folder and fill out the `human_evaluation_sheet.csv`.\n\n")
        f.write("### Scoring Criteria (1: Worst, 5: Best)\n")
        f.write("- **Human_Subject_Consistency:** Is the main subject identical across shots without morphing?\n")
        f.write("- **Human_Background_Dynamics:** Does the background properly change (Track S) or stay stable (Track M)?\n")
        f.write("- **Human_Prompt_Adherence:** Does the video actually follow the action/camera instructions?\n")
        f.write("- **Human_Cut_Sharpness:** Is the transition a sharp cinematic cut (5) or a messy fade/morph (1)?\n")
        f.write("- **Notes_or_Amnesia_Observed:** Mark 'O' if the subject's identity completely breaks or fuses with background.\n\n")
        f.write("After completion, we will use this CSV to compute Pearson/Spearman correlation with our DSA and Decoupled metrics to validate human alignment.\n")

    print(f"✅ Human Evaluation set up perfectly! Check {EVAL_DIR}")

if __name__ == "__main__":
    main()
