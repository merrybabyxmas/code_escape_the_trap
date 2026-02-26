import sys
import os
import json
import subprocess
from pathlib import Path
import torch
import gc
from PIL import Image
import numpy as np
import imageio

from diffusers import (
    AnimateDiffPipeline, 
    CogVideoXPipeline, 
    LTXPipeline,
    StableVideoDiffusionPipeline,
    TextToVideoSDPipeline,
    MotionAdapter,
    TextToVideoZeroPipeline
)
from diffusers.utils import export_to_video

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_JSON = os.path.join(SCRIPT_DIR, "datasets/dynamic_msv_bench.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def run_model(model_name, scenarios):
    print(f"🚀 Generating with {model_name}...")
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        if model_name == "CogVideoX":
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=" ".join(s['prompts']), num_frames=16).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "LTX-Video":
            pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=" ".join(s['prompts']), num_frames=32).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "SVD":
            pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16").to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                img = Image.new('RGB', (512, 512), color=(100, 100, 100))
                res = pipe(img, decode_chunk_size=8).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "AnimateDiff":
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=s['prompts'][0], num_frames=16).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "ModelScope":
            pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=" ".join(s['prompts']), num_frames=16).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "StoryDiffusion":
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                frames = []
                for p in s['prompts']:
                    res = pipe(prompt=f"{s['subject']}, {p}", num_frames=8).frames[0]
                    frames.extend(res)
                export_to_video(frames, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "FreeNoise":
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=" Then ".join(s['prompts']), num_frames=32, guidance_scale=15.0).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
        elif model_name == "DirecT2V":
            pipe = TextToVideoZeroPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=". ".join(s['prompts']), video_length=8).images
                res = [(r * 255).astype("uint8") for r in res]
                imageio.mimsave(os.path.join(out_dir, f"{s['id']}.mp4"), res, fps=8)
            del pipe
        elif model_name == "Mora":
            pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16).to(DEVICE)
            for s in scenarios:
                if os.path.exists(os.path.join(out_dir, f"{s['id']}.mp4")): continue
                res = pipe(prompt=" ".join(s['prompts']), num_frames=24).frames[0]
                export_to_video(res, os.path.join(out_dir, f"{s['id']}.mp4"))
            del pipe
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
    flush()

def main():
    if not os.path.exists(BENCHMARK_JSON): return
    with open(BENCHMARK_JSON, 'r') as f: bench_data = json.load(f)
    
    models = ["CogVideoX", "LTX-Video", "SVD", "StoryDiffusion", "FreeNoise", "ModelScope", "AnimateDiff", "Mora", "DirecT2V"]
    
    for m in models:
        run_model(m, bench_data)
        
    print("✅ Full Two-Track baseline generation complete.")

if __name__ == "__main__":
    main()
