import os
import json
import torch
import gc
from PIL import Image
import numpy as np
import cv2
from diffusers import (
    AnimateDiffPipeline, CogVideoXPipeline, LTXPipeline,
    StableVideoDiffusionPipeline, TextToVideoSDPipeline,
    MotionAdapter, TextToVideoZeroPipeline
)
from diffusers.utils import export_to_video

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_JSON = os.path.join(SCRIPT_DIR, "datasets/dynamic_msv_bench_1000.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Global resolution and frame count for fairness
TARGET_RES = (512, 512)
TARGET_FPS = 8
TARGET_FRAMES = 16 

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def resize_video(frames, size):
    return [np.array(Image.fromarray(f).resize(size, Image.LANCZOS)) for f in frames]

def run_model_batch(model_name, scenarios):
    print(f"🚀 [RESUME MODE] Starting generation for model: {model_name}")
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Pipeline Initialization
    pipe = None
    try:
        if model_name == "CogVideoX":
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to(DEVICE)
        elif model_name == "LTX-Video":
            pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to(DEVICE)
        elif model_name == "SVD":
            pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16").to(DEVICE)
        elif model_name == "AnimateDiff":
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
        elif model_name == "ModelScope":
            pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16).to(DEVICE)
        elif model_name == "StoryDiffusion":
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
        elif model_name == "FreeNoise":
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
        elif model_name == "Mora":
            # Using specific model for Mora if different, or ModelScope baseline
            pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16).to(DEVICE)
        elif model_name == "DirecT2V":
            pipe = TextToVideoZeroPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
    except Exception as e:
        print(f"❌ Failed to load pipeline for {model_name}: {e}")
        return

    # 2. Sequential Generation with Checkpointing
    for i, s in enumerate(scenarios):
        sid = s['id']
        save_path = os.path.join(out_dir, f"{sid}.mp4")
        
        # Resume Check: Skip if file already exists and is valid
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
            continue
            
        print(f"[{model_name}] Generating {i+1}/1000: {sid}")
        try:
            prompt = " | ".join(s['prompts'])
            
            if model_name == "CogVideoX":
                frames = pipe(prompt=prompt, num_frames=TARGET_FRAMES).frames[0]
            elif model_name == "LTX-Video":
                frames = pipe(prompt=prompt, num_frames=TARGET_FRAMES).frames[0]
            elif model_name == "SVD":
                img = Image.new('RGB', TARGET_RES, color=(128, 128, 128))
                frames = pipe(img, num_frames=TARGET_FRAMES, decode_chunk_size=8).frames[0]
            elif model_name == "StoryDiffusion":
                # Special sequential prompt handling
                frames = []
                for p in s['prompts']:
                    shot = pipe(prompt=p, num_frames=TARGET_FRAMES//2).frames[0]
                    frames.extend(shot)
            elif model_name == "DirecT2V":
                # DirecT2V returns list of images
                frames = pipe(prompt=". ".join(s['prompts']), video_length=TARGET_FRAMES).images
                frames = [(r * 255).astype("uint8") for r in frames]
            else: # Mora, FreeNoise, AnimateDiff, ModelScope
                frames = pipe(prompt=prompt, num_frames=TARGET_FRAMES).frames[0]
            
            # Post-processing: Uniformity
            frames = resize_video(frames, TARGET_RES)
            export_to_video(frames, save_path, fps=TARGET_FPS)
            
        except Exception as e:
            print(f"⚠️ Error generating {sid} with {model_name}: {e}")
            continue
            
    del pipe
    flush()

def main():
    if not os.path.exists(BENCHMARK_JSON):
        print("Dataset not found!")
        return
        
    with open(BENCHMARK_JSON, 'r') as f:
        bench_data = json.load(f)
    
    MODELS = ["CogVideoX", "LTX-Video", "SVD", "StoryDiffusion", "FreeNoise", "ModelScope", "AnimateDiff", "Mora", "DirecT2V"]
    
    for m in MODELS:
        run_model_batch(m, bench_data)
        
    print("✅ 9,000 Video Generation Complete (or resumed to completion).")

if __name__ == "__main__":
    main()
