import lpips
import torch
import numpy as np

class LpipsEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)

    def calculate_frame_by_frame(self, frames):
        """
        Calculates LPIPS distances between all adjacent frames.
        Returns a list of distances.
        """
        if len(frames) < 2: return []
        
        distances = []
        for i in range(len(frames) - 1):
            with torch.no_grad():
                d = self.loss_fn(frames[i].to(self.device), frames[i+1].to(self.device))
                distances.append(float(d.item()))
        return distances

    def calculate_cut_sharpness(self, frames):
        """
        Calculates the Cut Sharpness using a Dynamic t_cut detection algorithm.
        Instead of assuming a fixed cut frame, it calculates the 'Peak Prominence'
        of LPIPS distances across all adjacent frames to differentiate a clean
        hard cut from a gradual fade/morphing.
        
        frames: List of torch Tensors (C, H, W) normalized to [-1, 1]
        """
        if len(frames) < 2: return 0.0
        
        distances = []
        for i in range(len(frames) - 1):
            with torch.no_grad():
                d = self.loss_fn(frames[i].to(self.device), frames[i+1].to(self.device))
                distances.append(d.item())
        
        distances = np.array(distances)
        
        # Dynamic t_cut Detection: The true cut happens at the peak distance.
        max_dist = np.max(distances)
        
        # Calculate Peak Prominence: How sharp is the cut compared to background motion?
        # A sharp cut has a very high peak relative to neighbors (mean distance).
        mean_dist = np.mean(distances)
        
        # Prominence ratio. We use log scale to bound it nicely, or a simple ratio.
        # Adding epsilon to prevent division by zero.
        prominence = max_dist / (mean_dist + 1e-5)
        
        # Normalize to a 0-1 scale intuitively: 
        # If max_dist == mean_dist (no cut, just uniform motion), score -> 0
        # If max_dist >> mean_dist (sharp cut), score -> 1
        sharpness = 1.0 - (1.0 / (prominence + 1.0))
        
        return float(sharpness)
