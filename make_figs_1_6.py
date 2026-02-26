import os
import torch
import cv2
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import StableDiffusionPipeline
import clip

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/home/dongwoo43/paper_escapethetrap/escapethetrap/outputs"
FIGURES_DIR = "/home/dongwoo43/paper_escapethetrap/paper_escapethetrap/Escaping-the-Static-Video-Trap/figures"
BENCHMARK_JSON = "/home/dongwoo43/paper_escapethetrap/escapethetrap/datasets/dynamic_msv_bench.json"
# ---------------------

def extract_representative_frames(video_path, num_shots=4):
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0: return None
    shot_length = len(frames) // num_shots
    if shot_length == 0: shot_length = 1
    rep_frames = [frames[min(i*shot_length + shot_length//2, len(frames)-1)] for i in range(num_shots)]
    return [Image.fromarray(f) for f in rep_frames]

def make_figure_1():
    print("Generating Figure 1 (Teaser) with Owl Example...")
    # Use set_a_002 (Clockwork Owl)
    sid = "set_a_002"
    with open(BENCHMARK_JSON, 'r') as f:
        bench_data = json.load(f)
    scenario = next(s for s in bench_data if s['id'] == sid)
    prompts = scenario['prompts']
    
    frames_a = extract_representative_frames(os.path.join(OUTPUT_DIR, "CogVideoX", f"{sid}.mp4"))
    frames_b = extract_representative_frames(os.path.join(OUTPUT_DIR, "StoryDiffusion", f"{sid}.mp4"))
    
    if not frames_a or not frames_b:
        print(f"Videos for {sid} not found!")
        return

    # Ideal Row
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
    frames_c = []
    for p in prompts:
        gen = torch.Generator(DEVICE).manual_seed(2026)
        img = pipe(prompt=p + ", detailed clockwork owl, masterpiece", generator=gen, num_inference_steps=20).images[0]
        frames_c.append(img.resize((frames_a[0].width, frames_a[0].height)))
    del pipe; torch.cuda.empty_cache()

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
    locations = ["Rainforest", "Space", "Cyberpunk City", "Underwater"]
    for i, loc in enumerate(locations):
        axes[0, i].set_title(f"Shot {i+1}: {loc}", fontsize=14, pad=10)
    
    rows = [
        (frames_a, "Model A: CogVideoX\n(The Static Trap)\nDSA $\\approx$ 0.00"),
        (frames_b, "Model B: StoryDiffusion\n(Identity Amnesia)\nDSA $\\approx$ 0.25"),
        (frames_c, "Ideal Target\n(Decoupled Dynamics)\nDSA $\\rightarrow$ 1.00")
    ]
    
    for r, (frames, label) in enumerate(rows):
        for c in range(4):
            axes[r, c].imshow(frames[c])
            axes[r, c].axis('off')
            if c == 0:
                axes[r, c].text(-0.15, 0.5, label, fontsize=14, ha='right', va='center', rotation=0, transform=axes[r,c].transAxes, fontweight='bold')
    
    out_path = os.path.join(FIGURES_DIR, "fig1_teaser_real.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Figure 1 (Owl) saved to {out_path}")

def make_figure_6():
    # Heatmaps logic remains same as it uses representative probability matrix
    print("Generating Figure 6 (Heatmaps)...")
    sid = "set_a_002"
    frames_a = extract_representative_frames(os.path.join(OUTPUT_DIR, "CogVideoX", f"{sid}.mp4"))
    frames_b = extract_representative_frames(os.path.join(OUTPUT_DIR, "StoryDiffusion", f"{sid}.mp4"))
    
    if not frames_a or not frames_b: return

    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    with open(BENCHMARK_JSON, 'r') as f: bench_data = json.load(f)
    prompts = next(s for s in bench_data if s['id'] == sid)['prompts']

    def get_prob_matrix(frames):
        images = torch.stack([preprocess(img) for img in frames]).to(DEVICE)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        texts = clip.tokenize(prompts, truncate=True).to(DEVICE)
        text_features = model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        raw_sim_matrix = (image_features @ text_features.t()) 
        tau = model.logit_scale.exp().item()
        prob_matrix = torch.softmax(tau * raw_sim_matrix, dim=0)
        return prob_matrix.detach().cpu().numpy()

    mat_a = get_prob_matrix(frames_a)
    mat_b = get_prob_matrix(frames_b)
    mat_c = np.eye(4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = "YlGnBu"
    labels = [f"P{i+1}" for i in range(4)]
    shots = [f"S{i+1}" for i in range(4)]

    sns.heatmap(mat_a, annot=True, fmt=".2f", cmap=cmap, cbar=False, ax=axes[0], xticklabels=labels, yticklabels=shots, vmin=0, vmax=1)
    axes[0].set_title("(A) Static Trap\nUniform Confusion", fontsize=15)
    sns.heatmap(mat_b, annot=True, fmt=".2f", cmap=cmap, cbar=False, ax=axes[1], xticklabels=labels, yticklabels=shots, vmin=0, vmax=1)
    axes[1].set_title("(B) Amnesia\nPrompt Bleeding", fontsize=15)
    sns.heatmap(mat_c, annot=True, fmt=".2f", cmap=cmap, cbar=True, ax=axes[2], xticklabels=labels, yticklabels=shots, vmin=0, vmax=1)
    axes[2].set_title("(C) Ideal Model\nPerfect Alignment", fontsize=15)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig6_heatmaps_real.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Figure 6 saved to {out_path}")

if __name__ == "__main__":
    make_figure_1()
    make_figure_6()
