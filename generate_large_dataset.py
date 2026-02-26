import json
import os
import random

def generate_bench():
    subjects = [
        "A Victorian-era detective", "A cyberpunk courier with bionic arms", "A majestic phoenix",
        "A curious toddler in a yellow raincoat", "A futuristic combat drone", "A medieval knight in shining armor",
        "A wise old turtle with a mossy shell", "A floating crystal jellyfish", "A red panda wearing a tiny backpack",
        "A giant robot gardener", "A ballerina made of starlight", "A street photographer in Tokyo",
        "A snow leopard in a tuxedo", "A vintage steampunk airship", "A mysterious wizard with a staff of ice"
    ]
    
    locations = [
        "a dense Amazonian rainforest", "the neon-lit rooftops of a futuristic Seoul", "a tranquil Zen garden in Kyoto",
        "the crater of an active volcano", "a crowded marketplace in ancient Rome", "the silent corridors of a space station",
        "an underwater city made of coral", "a floating island in the clouds", "a dusty library with infinite shelves",
        "the surface of a frozen moon", "a vibrant coral reef", "a post-apocalyptic desert wasteland"
    ]
    
    actions = [
        "is running through", "is dancing in", "is searching for something in", "is floating slowly above",
        "is performing a magic ritual in", "is repairing a broken machine in", "is playing a golden flute in",
        "is fighting an invisible enemy in", "is paintng a masterpiece in", "is resting peacefully in"
    ]

    axes = ["background_shift", "action_change", "object_interaction", "camera_motion"]
    
    scenarios = []
    
    # Generate 1,000 scenarios
    for i in range(1000):
        axis = random.choice(axes)
        subject = random.choice(subjects)
        
        prompts = []
        if axis == "background_shift":
            # Subject remains same, location changes drastically
            locs = random.sample(locations, 4)
            for loc in locs:
                prompts.append(f"{subject} {random.choice(actions)} {loc}.")
        
        elif axis == "action_change":
            # Subject and location same, action changes
            loc = random.choice(locations)
            acts = random.sample(actions, 4)
            for act in acts:
                prompts.append(f"{subject} {act} {loc}.")
                
        elif axis == "object_interaction":
            loc = random.choice(locations)
            objs = ["a glowing orb", "a mysterious ancient scroll", "a holographic map", "a mechanical butterfly"]
            for obj in objs:
                prompts.append(f"{subject} interacting with {obj} in {loc}.")
                
        elif axis == "camera_motion":
            loc = random.choice(locations)
            act = random.choice(actions)
            motions = ["panning slowly across", "zooming in on", "tracking from a low angle", "tilting up towards"]
            for motion in motions:
                prompts.append(f"A {motion} {subject} as it {act} {loc}.")
        
        scenarios.append({
            "id": f"{axis[:3]}_{i:04d}",
            "axis": axis,
            "subject": subject,
            "prompts": prompts
        })
        
    return scenarios

def main():
    print("🚀 Generating 1,000 high-diversity benchmark scenarios...")
    bench_data = generate_bench()
    
    output_path = "datasets/dynamic_msv_bench.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(bench_data, f, indent=4)
        
    print(f"✅ Successfully created {len(bench_data)} scenarios in {output_path}")

if __name__ == "__main__":
    main()
