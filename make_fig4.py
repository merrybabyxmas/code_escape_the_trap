import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_frames(video_path, num_frames=4):
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if len(frames) == 0: return None
    
    # Extract evenly spaced frames
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    return [frames[i] for i in indices]

def make_filmstrip():
    model1 = "CogVideoX" # Static Trap
    model2 = "AnimateDiff" # Dynamics but Amnesia/Morphing
    
    vid1 = f"outputs/{model1}/set_b_000.mp4"
    vid2 = f"outputs/{model2}/set_b_000.mp4"
    
    frames1 = get_frames(vid1)
    frames2 = get_frames(vid2)
    
    if not frames1 or not frames2:
        print("Videos not found for filmstrip.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        axes[0, i].imshow(frames1[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(frames2[i])
        axes[1, i].axis('off')
        
    axes[0, 0].set_title("Shot 1: Panning", fontsize=12)
    axes[0, 1].set_title("Shot 2: Zooming", fontsize=12)
    axes[0, 2].set_title("Shot 3: Tracking", fontsize=12)
    axes[0, 3].set_title("Shot 4: Rotating", fontsize=12)
    
    plt.text(-0.1, 0.5, 'CogVideoX\n(Static Trap)', fontsize=14, ha='center', va='center', rotation=90, transform=axes[0,0].transAxes)
    plt.text(-0.1, 0.5, 'AnimateDiff\n(High Motion)', fontsize=14, ha='center', va='center', rotation=90, transform=axes[1,0].transAxes)

    plt.tight_layout()
    out_path = "/home/dongwoo43/paper_escapethetrap/paper_escapethetrap/Escaping-the-Static-Video-Trap/figures/fig4_filmstrip.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Filmstrip saved to {out_path}")

if __name__ == "__main__":
    os.chdir("/home/dongwoo43/paper_escapethetrap/escapethetrap")
    make_filmstrip()
