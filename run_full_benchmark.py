import os, torch, json, sys, imageio, numpy as np
from diffusers import (
    AnimateDiffPipeline, MotionAdapter, CogVideoXPipeline, LTXPipeline, 
    StableVideoDiffusionPipeline, TextToVideoSDPipeline, TextToVideoZeroPipeline
)
from diffusers.utils import export_to_video
from PIL import Image
import datetime
import argparse

# Ensure scripts dir is in path for evaluation import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from evaluate_metrics import evaluate_single_video

os.chdir(SCRIPT_DIR)
bench = json.load(open("datasets/dynamic_msv_bench.json"))
DEVICE = "cuda"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open("python_progress.log", "a", buffering=1) as f:
        f.write(formatted_msg + "\n")

def run_all(target_model=None):
    log(f"🚀 Starting Unified Benchmark Run (100 scenarios)")
    all_models = ["CogVideoX", "LTX-Video", "SVD", "StoryDiffusion", "FreeNoise", "ModelScope", "AnimateDiff", "Mora", "DirecT2V"]
    
    if target_model:
        models = [target_model]
    else:
        models = all_models

    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

    for mname in models:
        log(f"📦 Processing Model: {mname}")
        out_dir = f"outputs/{mname}"
        os.makedirs(out_dir, exist_ok=True)
        
        for s in bench:
            sid = s['id']
            out_path = f"{out_dir}/{sid}.mp4"
            
            # Check if we already have the metrics for this sample
            per_sample_path = os.path.join(SCRIPT_DIR, "outputs/per_sample_results_full.json")
            if os.path.exists(per_sample_path):
                with open(per_sample_path, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        if mname in existing_data and sid in existing_data[mname]:
                            continue # Skip generation if metrics already exist
                    except: pass

            if not os.path.exists(out_path):
                log(f"🎬 Generating {sid}...")
                try:
                    if mname == "CogVideoX":
                        pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to(DEVICE)
                        res = pipe(prompt=" ".join(s["prompts"]), num_frames=16).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "LTX-Video":
                        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to(DEVICE)
                        res = pipe(prompt=" ".join(s["prompts"]), num_frames=16).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "SVD":
                        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16").to(DEVICE)
                        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
                        res = pipe(img, decode_chunk_size=8).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "StoryDiffusion":
                        pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
                        frames = []
                        generator = torch.Generator(DEVICE).manual_seed(42)
                        for p in s["prompts"]:
                            res = pipe(prompt=f"{s['subject']}, {p}, consistent", num_frames=8, generator=generator).frames[0]
                            frames.extend(res)
                        export_to_video(frames, out_path)
                        del pipe
                    elif mname == "FreeNoise":
                        pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
                        res = pipe(prompt=" Then ".join(s["prompts"]), num_frames=32, guidance_scale=15.0).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "ModelScope":
                        pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16).to(DEVICE)
                        res = pipe(prompt=" ".join(s["prompts"]), num_frames=16).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "AnimateDiff":
                        pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
                        res = pipe(prompt=s["prompts"][0], num_frames=16).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "Mora":
                        pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16).to(DEVICE)
                        res = pipe(prompt=" ".join(s["prompts"]), num_frames=24).frames[0]
                        export_to_video(res, out_path)
                        del pipe
                    elif mname == "DirecT2V":
                        pipe = TextToVideoZeroPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
                        res = pipe(prompt=". ".join(s["prompts"]), video_length=8).images
                        res = [(r * 255).astype("uint8") for r in res]
                        imageio.mimsave(out_path, res, fps=8)
                        del pipe
                    
                    torch.cuda.empty_cache()
                except Exception as e:
                    log(f"   ❌ {mname} failed on {sid}: {e}")
                    continue

            # 📊 Real-time evaluation after generation
            if os.path.exists(out_path):
                log(f"📊 Evaluating {sid}...")
                try:
                    evaluate_single_video(mname, s)
                except Exception as eval_e:
                    log(f"   ❌ Evaluation failed for {sid}: {eval_e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    run_all(target_model=args.model)
