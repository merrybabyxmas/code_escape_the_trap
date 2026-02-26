import os
import json
import random
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(DATASET_DIR, "dynamic_msv_bench_1000.json")

# --- Taxonomy Vocabularies ---
SUBJECTS = [
    ("a retro-futuristic robot", "Inanimate"), ("a brave stray dog", "Animal"),
    ("a cybernetic samurai", "Humanoid"), ("a glowing crystal artifact", "Inanimate"),
    ("a wise old wizard", "Humanoid"), ("a majestic white tiger", "Animal"),
    ("an autonomous exploration rover", "Inanimate"), ("a mysterious space traveler", "Humanoid")
]

# Track S: Semantic Leap (Backgrounds)
SPATIAL = [("dense jungle", "deep space"), ("underwater city", "martian surface"), ("bustling metropolis", "quiet desert oasis")]
TEMPORAL = [("medieval castle", "cyberpunk megacity"), ("ancient rome", "futuristic utopia"), ("prehistoric cave", "modern laboratory")]
ATMOSPHERE = [("sunny day", "raging blizzard"), ("calm evening", "volcanic eruption"), ("clear sky", "acid rain storm")]
SCALE = [("desk top", "planetary orbit"), ("living room floor", "galaxy overview"), ("microscopic level", "macroscopic city view")]

# Track M: Motion Continuity (Camera Motions)
TRANSLATIONAL = ["panning left", "panning right", "tilting up", "tilting down"]
DEPTH = ["zooming in", "extreme close-up", "zooming out", "pulling back"]
ORBIT = ["orbiting 360 degrees", "tracking subject from the side", "revolving around the subject"]
COMPOUND = ["panning right while zooming in", "tilting down while zooming out", "orbiting while pulling back"]

def generate_prompts():
    dataset = []
    scenario_id = 1
    
    # 1. Generate Track S (500 Prompts)
    track_s_categories = [
        ("Spatial", SPATIAL, "Hard", "Identity Amnesia"),
        ("Temporal", TEMPORAL, "Hard", "Identity Amnesia"),
        ("Atmosphere", ATMOSPHERE, "Medium", "Static Trap"),
        ("Scale", SCALE, "Hard", "Identity Amnesia")
    ]
    
    for cat_name, shifts, diff, vuln in track_s_categories:
        count = 0
        while count < 125:
            subj, subj_type = random.choice(SUBJECTS)
            shift = random.choice(shifts)
            
            prompt1 = f"{subj} standing in a {shift[0]}, cinematic lighting, high quality."
            prompt2 = f"{subj} standing in a {shift[1]}, cinematic lighting, high quality."
            
            dataset.append({
                "id": f"track_s_{scenario_id:04d}",
                "track": "Track S (Semantic Leap)",
                "sub_category": cat_name,
                "metadata": {
                    "subject_type": subj_type,
                    "difficulty": diff,
                    "target_vulnerability": vuln
                },
                "prompts": [prompt1, prompt2]
            })
            scenario_id += 1
            count += 1

    # 2. Generate Track M (500 Prompts)
    track_m_categories = [
        ("Translational", TRANSLATIONAL, "Medium", "Static Trap"),
        ("Depth", DEPTH, "Hard", "Identity Amnesia"),
        ("Tracking_Orbit", ORBIT, "Hard", "Background Hallucination"),
        ("Compound", COMPOUND, "Hard", "Context Bleeding")
    ]
    
    bg_list = ["quiet forest", "neon city street", "abandoned space station"]
    
    for cat_name, motions, diff, vuln in track_m_categories:
        count = 0
        while count < 125:
            subj, subj_type = random.choice(SUBJECTS)
            bg = random.choice(bg_list)
            motion = random.choice(motions)
            
            prompt1 = f"{subj} standing still in a {bg}, stationary camera, high quality."
            prompt2 = f"{subj} standing still in a {bg}, camera {motion}, high quality."
            
            dataset.append({
                "id": f"track_m_{scenario_id:04d}",
                "track": "Track M (Motion Continuity)",
                "sub_category": cat_name,
                "metadata": {
                    "subject_type": subj_type,
                    "difficulty": diff,
                    "target_vulnerability": vuln
                },
                "prompts": [prompt1, prompt2]
            })
            scenario_id += 1
            count += 1
            
    return dataset

def main():
    dataset = generate_prompts()
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
    print(f"✅ Generated deeply categorized dataset with {len(dataset)} prompts.")
    print(f"✅ Saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
