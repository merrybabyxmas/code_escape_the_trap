import cv2
import numpy as np
import os
import torch
from PIL import Image
import sys

# Ensure core module is accessible
sys.path.append("/home/dongwoo43/paper/escapethetrap")
from core.metrics.dino_eval import DinoEvaluator
from core.metrics.clip_eval import ClipEvaluator

vid_path = "/home/dongwoo43/paper/escapethetrap/outputs/AnimateDiff/cam_0000.mp4"
if not os.path.exists(vid_path):
    print("Video not found")
    sys.exit()

cap = cv2.VideoCapture(vid_path)
ret, frame = cap.read()
if not ret:
    print("Failed to read frame")
else:
    print(f"Frame shape: {frame.shape}, max value: {np.max(frame)}")

# Test evaluators
dino = DinoEvaluator()
clip_eval = ClipEvaluator()

frames = []
for i in range(4):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * 4)
    ret, f = cap.read()
    if ret:
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

if frames:
    subj = dino.calculate_subject_consistency(frames)
    print(f"DINO Subject Consistency: {subj}")
    
    prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
    diag = clip_eval.calculate_diagonal_alignment(frames, prompts)
    print(f"CLIP Diagonal Alignment: {diag}")
cap.release()
