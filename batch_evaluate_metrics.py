import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import sys

# Ensure core module is accessible
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from core.metrics.dino_eval import DinoEvaluator
from core.metrics.clip_eval import ClipEvaluator
from core.metrics.lpips_eval import LpipsEvaluator

# --- Configuration ---
BENCHMARK_JSON = os.path.join(SCRIPT_DIR, "datasets/dynamic_msv_bench_1000.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
PER_SAMPLE_RESULTS_PATH = os.path.join(OUTPUT_DIR, "per_sample_results_huge.json")

# Global Evaluators
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
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    if not os.path.exists(BENCHMARK_JSON): return
    with open(BENCHMARK_JSON, 'r') as f: bench_data = json.load(f)
    
    dino, clip_eval, lpips = get_evaluators()
    per_sample = load_json(PER_SAMPLE_RESULTS_PATH)
    
    MODELS = ["CogVideoX", "LTX-Video", "SVD", "StoryDiffusion", "FreeNoise", "ModelScope", "AnimateDiff", "Mora", "DirecT2V"]
    
    for model_name in MODELS:
        print(f"📊 Evaluating {model_name}...")
        if model_name not in per_sample: per_sample[model_name] = {}
        
        for i, s in enumerate(bench_data):
            sid = s['id']
            # Resume Check
            if sid in per_sample[model_name]: continue
            
            vid_path = os.path.join(OUTPUT_DIR, model_name, f"{sid}.mp4")
            if not os.path.exists(vid_path): continue
            
            print(f"[{model_name}] Analyzing {i+1}/1000: {sid}")
            frames = extract_frames(vid_path)
            if len(frames) < 2: continue
            
            # Metric Calculation
            num_shots = len(s['prompts'])
            shot_len = len(frames) // num_shots
            rep_frames_pil = [Image.fromarray(frames[min(j*shot_len + shot_len//2, len(frames)-1)]) for j in range(num_shots)]
            
            s_score = dino.calculate_subject_consistency(rep_frames_pil)
            
            # Background Diversity
            full_embs = torch.cat([dino.get_embedding(img) for img in rep_frames_pil])
            b_score = 1.0 - torch.mm(full_embs, full_embs.t()).mean().item()
            
            d_score = clip_eval.calculate_diagonal_alignment(rep_frames_pil, s['prompts'])
            
            # LPIPS conversion for frames
            import torchvision.transforms as T
            t = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            c_score = lpips.calculate_cut_sharpness([t(Image.fromarray(f)) for f in frames])
            
            per_sample[model_name][sid] = {
                'metrics': {'subj': s_score, 'bg': b_score, 'diag': d_score, 'cut': c_score},
                'track': s['track'],
                'sub_category': s['sub_category']
            }
            
            # Periodically save to avoid data loss
            if i % 50 == 0:
                save_json(per_sample, PER_SAMPLE_RESULTS_PATH)
                
        save_json(per_sample, PER_SAMPLE_RESULTS_PATH)

    print(f"✅ All 7,000 videos evaluated and results saved to {PER_SAMPLE_RESULTS_PATH}")

if __name__ == "__main__":
    main()
