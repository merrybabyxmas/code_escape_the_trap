import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
import sys
import shutil

# Ensure core module is accessible
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from core.metrics.dino_eval import DinoEvaluator
from core.metrics.clip_eval import ClipEvaluator
from core.metrics.lpips_eval import LpipsEvaluator

# --- Configuration ---
BENCHMARK_JSON = os.path.join(SCRIPT_DIR, "datasets/dynamic_msv_bench.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
EXAMPLE_DIR = os.path.join(OUTPUT_DIR, "examples")
PER_SAMPLE_RESULTS_PATH = os.path.join(OUTPUT_DIR, "per_sample_results_full.json")

# Global Evaluators (Initialized lazily)
evaluators = {"dino": None, "clip": None, "lpips": None}

def get_evaluators():
    if evaluators["dino"] is None:
        print("Initializing metric models...")
        evaluators["dino"] = DinoEvaluator()
        evaluators["clip"] = ClipEvaluator()
        evaluators["lpips"] = LpipsEvaluator()
    return evaluators["dino"], evaluators["clip"], evaluators["lpips"]

def extract_frames(video_path):
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def load_json(path, default={}):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try: return json.load(f)
            except: return default
    return default

def save_json(data, path):
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)

def evaluate_single_video(model_name, scenario):
    sid = scenario['id']
    prompts = scenario['prompts']
    vid_path = os.path.join(OUTPUT_DIR, model_name, f"{sid}.mp4")
    
    if not os.path.exists(vid_path): return None
    
    frames = extract_frames(vid_path)
    if len(frames) < 4: return None
    
    dino, clip_eval, lpips = get_evaluators()
    
    num_shots = len(prompts)
    shot_length = len(frames) // num_shots
    if shot_length == 0: shot_length = 1
    rep_frames_pil = [Image.fromarray(frames[min(i*shot_length + shot_length//2, len(frames)-1)]) for i in range(num_shots)]
    
    s = dino.calculate_subject_consistency(rep_frames_pil)
    full_embs = torch.cat([dino.get_embedding(img) for img in rep_frames_pil])
    b = 1.0 - torch.mm(full_embs, full_embs.t()).mean().item()
    d = clip_eval.calculate_diagonal_alignment(rep_frames_pil, prompts)
    t = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    c = lpips.calculate_cut_sharpness([t(Image.fromarray(f)) for f in frames[::4]])
    
    result = {
        'prompts': prompts,
        'metrics': {'subj': s, 'bg': b, 'diag': d, 'cut': c}
    }
    
    # Incremental update to master JSON
    per_sample = load_json(PER_SAMPLE_RESULTS_PATH)
    if model_name not in per_sample: per_sample[model_name] = {}
    per_sample[model_name][sid] = result
    save_json(per_sample, PER_SAMPLE_RESULTS_PATH)
    
    return result

def main():
    if not os.path.exists(BENCHMARK_JSON): return
    with open(BENCHMARK_JSON, 'r') as f: bench_data = json.load(f)
    
    get_evaluators()
    
    results_set_a = {}
    results_set_b = {}
    
    # We'll just run summary based on existing per_sample_results_full.json
    per_sample = load_json(PER_SAMPLE_RESULTS_PATH)
    
    MODELS = ["CogVideoX", "LTX-Video", "SVD", "StoryDiffusion", "FreeNoise", "ModelScope", "AnimateDiff", "Mora", "DirecT2V"]
    
    for model_name in MODELS:
        if model_name not in per_sample: continue
        
        metrics_a = {"subj": [], "bg": [], "diag": [], "cut": []}
        metrics_b = {"subj": [], "bg": [], "diag": [], "cut": []}
        
        for sid, entry in per_sample[model_name].items():
            # Find scenario type
            stype = "semantic_shift" if "set_a" in sid else "motion_shift"
            m = entry['metrics']
            if stype == "semantic_shift":
                metrics_a["subj"].append(m['subj']); metrics_a["bg"].append(m['bg']); metrics_a["diag"].append(m['diag']); metrics_a["cut"].append(m['cut'])
            else:
                metrics_b["subj"].append(m['subj']); metrics_b["bg"].append(m['bg']); metrics_b["diag"].append(m['diag']); metrics_b["cut"].append(m['cut'])
        
        if metrics_a["subj"]:
            results_set_a[model_name] = {
                "Subject_Consistency": float(np.mean(metrics_a["subj"])),
                "Background_Diversity": float(np.mean(metrics_a["bg"])),
                "Diagonal_Alignment": float(np.mean(metrics_a["diag"])),
                "Cut_Sharpness": float(np.mean(metrics_a["cut"]))
            }
        if metrics_b["subj"]:
            results_set_b[model_name] = {
                "Subject_Consistency": float(np.mean(metrics_b["subj"])),
                "Background_Diversity": float(np.mean(metrics_b["bg"])),
                "Diagonal_Alignment": float(np.mean(metrics_b["diag"])),
                "Cut_Sharpness": float(np.mean(metrics_b["cut"]))
            }

    save_json(results_set_a, os.path.join(OUTPUT_DIR, "final_metrics_results_set_a.json"))
    save_json(results_set_b, os.path.join(OUTPUT_DIR, "final_metrics_results_set_b.json"))
    print("✅ Finalized summaries from per-sample data.")

if __name__ == "__main__":
    main()
