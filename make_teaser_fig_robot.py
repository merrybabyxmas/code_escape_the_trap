import os
import torch
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGURES_DIR = "/home/dongwoo43/paper_escapethetrap/paper_escapethetrap/Escaping-the-Static-Video-Trap/figures"
IDEAL_IMG_PATH = "/home/dongwoo43/paper_escapethetrap/paper_escapethetrap/Escaping-the-Static-Video-Trap/figures/images/fig_ideal.png"
# ---------------------

def create_teaser_v2():
    print("Generating Refined Teaser Figure (Robot Example)...")
    
    # 1. Load Ideal Image and split into 4 shots
    ideal_full = Image.open(IDEAL_IMG_PATH)
    w, h = ideal_full.size
    shot_w = w // 4
    ideal_shots = [ideal_full.crop((i*shot_w, 0, (i+1)*shot_w, h)) for i in range(4)]
    
    # 2. Setup SD for generating Trap/Amnesia examples
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
    
    prompts = [
        "A retro-futuristic red spherical robot with glowing blue eyes in a dense rainforest, photorealistic, 8k",
        "A retro-futuristic red spherical robot with glowing blue eyes in deep outer space, photorealistic, 8k",
        "A retro-futuristic red spherical robot with glowing blue eyes in a cyberpunk city at night, photorealistic, 8k",
        "A retro-futuristic red spherical robot with glowing blue eyes deep underwater, photorealistic, 8k"
    ]

    # --- Row A: Static Trap (Same Jungle image repeated) ---
    print("Creating Row A: Static Trap...")
    # Use the first shot of the ideal (Jungle) as the base for the static trap
    # To make it look like a "model" result, I'll generate one Jungle frame
    gen = torch.Generator(DEVICE).manual_seed(123)
    jungle_frame = pipe(prompts[0], generator=gen, num_inference_steps=25).images[0].resize((shot_w, h))
    static_shots = [jungle_frame] * 4

    # --- Row B: Identity Amnesia (Correct backgrounds, morphing robot) ---
    print("Creating Row B: Identity Amnesia...")
    amnesia_shots = []
    # Vary the robot description or seeds significantly to show amnesia
    amnesia_prompts = [
        "A red spherical robot in a dense rainforest",
        "A blue cube-shaped robot in outer space",
        "A yellow humanoid robot in a cyberpunk city",
        "A green triangular robot deep underwater"
    ]
    for i, p in enumerate(amnesia_prompts):
        gen = torch.Generator(DEVICE).manual_seed(i * 100 + 42)
        img = pipe(p + ", cinematic, 8k", generator=gen, num_inference_steps=25).images[0].resize((shot_w, h))
        amnesia_shots.append(img)

    del pipe; torch.cuda.empty_cache()

    # 3. Plotting
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.02, hspace=0.25)
    
    titles = ["Shot 1: Jungle", "Shot 2: Space", "Shot 3: City", "Shot 4: Ocean"]
    for i, t in enumerate(titles):
        axes[0, i].set_title(t, fontsize=16, fontweight='bold', pad=10)
    
    rows = [
        (static_shots, "Model A: Static Trap\n(Instruction Ignored)\nDSA $\\approx$ 0.00"),
        (amnesia_shots, "Model B: Identity Amnesia\n(Subject Morphs)\nDSA $\\approx$ 0.30"),
        (ideal_shots, "Ideal Target\n(Decoupled Dynamics)\nDSA $\\rightarrow$ 1.00")
    ]
    
    for r, (shots, label) in enumerate(rows):
        for c in range(4):
            axes[r, c].imshow(shots[c])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            # Add colored borders to highlight rows
            color = 'red' if r == 0 else ('orange' if r == 1 else 'green')
            for spine in axes[r,c].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            
            if c == 0:
                axes[r, c].text(-0.1, 0.5, label, fontsize=18, ha='right', va='center', transform=axes[r,c].transAxes, fontweight='bold', color=color)

    out_path = os.path.join(FIGURES_DIR, "fig1_teaser_robot.pdf")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"✅ Figure 1 (Robot) saved to {out_path}")

if __name__ == "__main__":
    create_teaser_v2()
