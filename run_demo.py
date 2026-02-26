import os, torch, json, sys, imageio, numpy as np
from diffusers import (
    AnimateDiffPipeline, MotionAdapter, CogVideoXPipeline, LTXPipeline, 
    StableVideoDiffusionPipeline, TextToVideoSDPipeline, TextToVideoZeroPipeline
)
from diffusers.utils import export_to_video
from PIL import Image

os.chdir("/home/dongwoo43/paper_escapethetrap/escapethetrap")
bench = json.load(open("datasets/dynamic_msv_bench.json"))
# We assume bench[0] is semantic_shift_000 and bench[1] is motion_shift_000 based on my previous truncation
s_a = bench[0]; s_b = bench[1]
DEVICE = "cuda"

def run_all():
    models = ["CogVideoX", "LTX-Video", "SVD", "StoryDiffusion", "FreeNoise", "ModelScope", "AnimateDiff", "Mora", "DirecT2V"]
    
    # Pre-load shared components
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

    for mname in models:
        print(f"🚀 Processing Model: {mname}")
        out_dir = f"outputs/{mname}"
        os.makedirs(out_dir, exist_ok=True)
        
        for s in [s_a, s_b]:
            sid = s['id']
            out_path = f"{out_dir}/{sid}.mp4"
            if os.path.exists(out_path): continue
            
            print(f"   Generating {sid}...")
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
                print(f"   ❌ {mname} failed on {sid}: {e}")

run_all()
print("✅ ALL 9 MODELS GENERATED FOR BOTH SETS.")
