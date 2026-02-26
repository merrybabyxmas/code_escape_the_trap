import os, torch, sys, json, imageio, asyncio, numpy as np
from diffusers import (
    AnimateDiffPipeline, MotionAdapter, CogVideoXPipeline, LTXPipeline, 
    StableVideoDiffusionPipeline, TextToVideoSDPipeline, TextToVideoZeroPipeline
)
from diffusers.utils import export_to_video
from PIL import Image

os.chdir("/home/dongwoo43/paper_escapethetrap/escapethetrap")
bench = json.load(open("datasets/dynamic_msv_bench.json"))
DEVICE = "cuda"

def run_targets():
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    
    for scenario in bench:
        sid = scenario['id']
        prompts = scenario['prompts']
        subject = scenario['subject']
        
        # 1. StoryDiffusion Proxy
        print(f"🚀 StoryDiffusion: {sid}")
        out_path = f"outputs/StoryDiffusion/{sid}.mp4"
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
            frames = []
            generator = torch.Generator(DEVICE).manual_seed(42)
            for p in prompts:
                res = pipe(prompt=f"{subject}, {p}, consistent", num_frames=8, generator=generator).frames[0]
                frames.extend(res)
            export_to_video(frames, out_path)
            del pipe; torch.cuda.empty_cache()

        # 2. FreeNoise Proxy
        print(f"🚀 FreeNoise: {sid}")
        out_path = f"outputs/FreeNoise/{sid}.mp4"
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16).to(DEVICE)
            res = pipe(prompt=" Then ".join(prompts), num_frames=32, guidance_scale=15.0).frames[0]
            export_to_video(res, out_path)
            del pipe; torch.cuda.empty_cache()

        # 3. CogVideoX
        print(f"🚀 CogVideoX: {sid}")
        out_path = f"outputs/CogVideoX/{sid}.mp4"
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to(DEVICE)
            res = pipe(prompt=" ".join(prompts), num_frames=16).frames[0]
            export_to_video(res, out_path)
            del pipe; torch.cuda.empty_cache()

        # 4. LTX
        print(f"🚀 LTX: {sid}")
        out_path = f"outputs/LTX-Video/{sid}.mp4"
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to(DEVICE)
            res = pipe(prompt=" ".join(prompts), num_frames=32).frames[0]
            export_to_video(res, out_path)
            del pipe; torch.cuda.empty_cache()

        # 5. SVD
        print(f"🚀 SVD: {sid}")
        out_path = f"outputs/SVD/{sid}.mp4"
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16").to(DEVICE)
            img = Image.new("RGB", (512, 512), color=(100, 100, 100))
            res = pipe(img, decode_chunk_size=8).frames[0]
            export_to_video(res, out_path)
            del pipe; torch.cuda.empty_cache()

run_targets()
print("✅ All specified scenarios for all models completed.")
