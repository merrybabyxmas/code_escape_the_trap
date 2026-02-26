import os
import json
import random

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "datasets/dynamic_msv_bench.json")
NUM_SCENARIOS_PER_SET = 50 
# ---------------------

def generate_semantic_shift_scenarios(num):
    subjects = ["A steampunk clockwork owl", "A glowing neon jellyfish", "A samurai in brass armor", "A golden retriever", "A robotic ballerina", "An obsidian dragon", "A robot gardener", "A cybernetic tiger", "A Victorian lady", "A futuristic soldier"]
    backgrounds = ["in rainforest", "in space", "in cyberpunk city", "underwater", "on snow mountain", "in cathedral", "on desert planet", "in high-tech lab", "in medieval village", "at a carnival"]
    scenarios = []
    for i in range(num):
        subject = random.choice(subjects)
        bgs = random.sample(backgrounds, 4)
        scenarios.append({
            "id": f"set_a_{i:03d}",
            "type": "semantic_shift",
            "subject": subject,
            "prompts": [f"{subject} {bg}" for bg in bgs]
        })
    return scenarios

def generate_motion_shift_scenarios(num):
    subjects = ["A medieval knight", "A futuristic hover-car", "A stone golem", "A detective", "A red sports car", "A hooded figure", "A bronze statue", "A sleek drone", "A crystal phoenix", "A pirate ship"]
    backgrounds = ["in courtyard", "on rainy bridge", "in serene forest", "in noir office", "on desert highway", "under waterfall", "in busy NYC", "on lunar pad", "in clock tower", "in mountain lake"]
    camera_motions = ["Panning slowly across", "Zooming in rapidly on", "Tracking low angle towards", "Rotating 360 degrees around"]
    scenarios = []
    for i in range(num):
        idx = i % len(subjects)
        subject, bg = subjects[idx], backgrounds[idx]
        scenarios.append({
            "id": f"set_b_{i:03d}",
            "type": "motion_shift",
            "subject": subject,
            "prompts": [f"{motion} {subject} {bg}" for motion in camera_motions]
        })
    return scenarios

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    scenarios = generate_semantic_shift_scenarios(NUM_SCENARIOS_PER_SET)
    scenarios.extend(generate_motion_shift_scenarios(NUM_SCENARIOS_PER_SET))
    with open(OUTPUT_PATH, 'w') as f: json.dump(scenarios, f, indent=4)
    print(f"✅ Generated {len(scenarios)} scenarios.")

if __name__ == "__main__":
    main()
