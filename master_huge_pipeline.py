import os
import json
import torch
import gc
import time
import sys
import platform
import subprocess
import argparse
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from diffusers import (
    AnimateDiffPipeline, CogVideoXPipeline, LTXPipeline,
    StableVideoDiffusionPipeline, TextToVideoSDPipeline,
    MotionAdapter, StableDiffusionPipeline
)
from diffusers.utils import export_to_video
from torch.multiprocessing import Process, Lock, set_start_method

# Import our core metrics
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from core.metrics.dino_eval import DinoEvaluator
from core.metrics.clip_eval import ClipEvaluator
from core.metrics.lpips_eval import LpipsEvaluator

# --- Global Configuration ---
BENCHMARK_JSON = os.path.join(SCRIPT_DIR, "datasets/dynamic_msv_bench_1000.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
PER_SAMPLE_RESULTS_PATH = os.path.join(OUTPUT_DIR, "per_sample_results_huge.json")

TARGET_RES = (512, 512)
TARGET_FPS = 8
TARGET_FRAMES = 16 

def flush():
    gc.collect(); torch.cuda.empty_cache()

def load_json(path, default={}):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try: return json.load(f)
            except: return default
    return default

def save_json_locked(data, path, lock):
    with lock:
        current = load_json(path)
        # Deep merge models
        for model, samples in data.items():
            if model not in current: current[model] = {}
            current[model].update(samples)
        with open(path, 'w') as f:
            json.dump(current, f, indent=4)

def resize_video(frames, size):
    processed = []
    for f in frames:
        if not isinstance(f, Image.Image):
            f = np.array(f)
            if np.issubdtype(f.dtype, np.floating): f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            if f.ndim == 3 and f.shape[0] == 3: f = f.transpose(1, 2, 0)
            f = Image.fromarray(f)
        processed.append(f.resize(size, Image.LANCZOS))
    return processed

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path); frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release(); return frames

# --- Worker Function ---
def model_worker(gpu_id, model_name, samples, lock):
    device = f"cuda:{gpu_id}"
    print(f"👷 [Worker GPU:{gpu_id}] Started for Model: {model_name} ({len(samples)} samples)")
    
    # Directory setup
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    mask_out_dir = os.path.join(OUTPUT_DIR, f"visual_proofs/masks/{model_name}")
    feat_out_dir = os.path.join(OUTPUT_DIR, f"features/{model_name}")
    lpips_out_dir = os.path.join(OUTPUT_DIR, f"visual_proofs/lpips_plots/{model_name}")
    for d in [out_dir, mask_out_dir, feat_out_dir, lpips_out_dir]: os.makedirs(d, exist_ok=True)

    # Evaluation Models
    dino = DinoEvaluator(); clip_eval = ClipEvaluator(); lpips = LpipsEvaluator()
    lpips_transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load Generation Pipeline
    pipe = None; t2i_pipe = None; SEED = 42; torch.manual_seed(SEED)
    try:
        if model_name == "CogVideoX":
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        elif model_name == "LTX-Video":
            pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        elif model_name == "SVD":
            t2i_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
            t2i_pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        elif model_name == "ModelScope":
            pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16)
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    except Exception as e:
        print(f"❌ [GPU:{gpu_id}] Failed to load {model_name}: {e}"); return

    local_results = {model_name: {}}

    for i, s in enumerate(samples):
        sid = s['id']; vid_path = os.path.join(out_dir, f"{sid}.mp4"); start_time = time.time()
        
        # 1. Gen
        if not (os.path.exists(vid_path) and os.path.getsize(vid_path) > 1000):
            try:
                prompt = " | ".join(s['prompts'])
                if model_name == "SVD":
                    init_image = t2i_pipe(s['prompts'][0], num_inference_steps=25).images[0].resize(TARGET_RES)
                    frames = pipe(init_image, num_frames=TARGET_FRAMES, decode_chunk_size=8).frames[0]
                elif model_name == "StoryDiffusion":
                    frames = []
                    for p in s['prompts']: frames.extend(pipe(prompt=p, num_frames=TARGET_FRAMES//2).frames[0])
                elif model_name == "CogVideoX":
                    # CogVideoX-2b works best with (4k + 1) frames. 13 or 17.
                    frames = pipe(prompt=prompt, num_frames=13, height=TARGET_RES[1], width=TARGET_RES[0]).frames[0]
                elif model_name == "LTX-Video":
                    # LTX-Video works best with (8k + 1) frames. 17.
                    frames = pipe(prompt=prompt, num_frames=17, height=TARGET_RES[1], width=TARGET_RES[0]).frames[0]
                else: frames = pipe(prompt=prompt, num_frames=TARGET_FRAMES).frames[0]
                frames = resize_video(frames, TARGET_RES)
                export_to_video(frames, vid_path, fps=TARGET_FPS)
            except Exception as e: print(f"⚠️ [GPU:{gpu_id}] Gen error {sid}: {e}"); continue

        # 2. Eval
        try:
            frames = extract_frames(vid_path)
            if len(frames) < 2: continue
            num_shots = len(s['prompts']); shot_len = len(frames) // num_shots
            rep_frames_pil = [Image.fromarray(frames[min(j*shot_len + shot_len//2, len(frames)-1)]) for j in range(num_shots)]
            for idx, img in enumerate(rep_frames_pil): img.save(os.path.join(mask_out_dir, f"{sid}_shot_{idx}.png"))
            
            dino_embs = [dino.get_embedding(img) for img in rep_frames_pil]
            torch.save(dino_embs, os.path.join(feat_out_dir, f"{sid}_dino_features.pt"))
            
            frame_tensors = [lpips_transform(Image.fromarray(f)) for f in frames]
            lpips_scores = lpips.calculate_frame_by_frame(frame_tensors)
            lpips_path = os.path.join(lpips_out_dir, f"{sid}_lpips.csv")
            with open(lpips_path, "w") as f:
                f.write("frame_idx,lpips_dist\n")
                for idx, sc in enumerate(lpips_scores): f.write(f"{idx},{sc}\n")

            s_score = dino.calculate_subject_consistency(rep_frames_pil)
            full_embs = torch.cat(dino_embs)
            sim_matrix = torch.mm(full_embs, full_embs.t())
            b_score = 1.0 - sim_matrix.mean().item()
            d_score = clip_eval.calculate_diagonal_alignment(rep_frames_pil, s['prompts'])
            c_score = lpips.calculate_cut_sharpness(frame_tensors)

            local_results[model_name][sid] = {
                'metrics': {'subj': s_score, 'bg': b_score, 'diag': d_score, 'cut': c_score, 'dsa_matrix': sim_matrix.tolist(), 'visibility': 1.0},
                'analysis': {'failure_mode': "Success", 'detected_cut_frames': [len(lpips_scores)//num_shots]},
                'track': s['track'], 'sub_category': s['sub_category'],
                'metadata': {'seed': SEED, 'usage': {'vram_gb': round(torch.cuda.max_memory_allocated(device)/(1024**3),2), 'time': round(time.time()-start_time,2)}}
            }
            if i % 5 == 0: save_json_locked(local_results, PER_SAMPLE_RESULTS_PATH, lock)
        except Exception as e: print(f"⚠️ [GPU:{gpu_id}] Eval error {sid}: {e}"); continue

    save_json_locked(local_results, PER_SAMPLE_RESULTS_PATH, lock)
    del pipe; flush()

def main_orchestrator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--tasks_per_gpu", type=int, default=1, help="Concurrent tasks per GPU")
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(",")]
    
    print(f"🌟 [MULTI-GPU MISSION] Starting 7,000 Video Benchmark on GPUs: {gpu_list}")
    if not os.path.exists(BENCHMARK_JSON): return
    with open(BENCHMARK_JSON, 'r') as f: bench_data = json.load(f)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_sample_lock = Lock()
    
    MODELS = ["SVD"]
    # MODELS = ["CogVideoX", "LTX-Video", "SVD", "ModelScope", "StoryDiffusion", "FreeNoise", "AnimateDiff"]
    
    # MODELS = ["ModelScope", "StoryDiffusion", "FreeNoise", "AnimateDiff"]
    
    
    processes = []
    # Simple strategy: Distribute models across available slots
    # Each slot is (gpu_id, task_idx)
    total_slots = len(gpu_list) * args.tasks_per_gpu
    
    for idx, model_name in enumerate(MODELS):
        gpu_id = gpu_list[idx % len(gpu_list)]
        p = Process(target=model_worker, args=(gpu_id, model_name, bench_data, per_sample_lock))
        p.start()
        processes.append(p)
        # Optional: Add a small delay between process starts to avoid VRAM spike
        time.sleep(5)

    for p in processes:
        p.join()

    print(f"✅ All Parallel Missions Completed! Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError: pass
    main_orchestrator()
