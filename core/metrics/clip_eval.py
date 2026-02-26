import torch
import clip
from PIL import Image

class ClipEvaluator:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    @torch.no_grad()
    def calculate_diagonal_alignment(self, video_shots, prompts, tau=None):
        """
        Diagonal Semantic Alignment (DSA) Metric
        1. Construction of Similarity Matrix M (K x K)
        2. Column-wise Softmax with Logit Scale (tau)
        3. Normalization: (MeanDiag - 1/K) / (1 - 1/K)
        
        Args:
            tau (float): Temperature hyperparameter. If None, uses CLIP's learned logit_scale.
        """
        K = len(video_shots)
        if K < 2: return 0.0

        # Extract Visual Features
        images = torch.stack([self.preprocess(img) for img in video_shots]).to(self.device)
        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Extract Text Features
        texts = clip.tokenize(prompts, truncate=True).to(self.device)
        text_features = self.model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 1. Similarity Matrix M
        # Row: Shots, Column: Prompts
        raw_sim_matrix = (image_features @ text_features.t()) 
        
        # 2. Column-wise Softmax with tau
        if tau is None:
            # tau is usually ~100.0 in CLIP
            tau = self.model.logit_scale.exp().item()
        
        # Column-wise Softmax: Distribution of shots for each prompt
        # sum(prob_matrix, dim=0) == [1, 1, ..., 1]
        prob_matrix = torch.softmax(tau * raw_sim_matrix, dim=0)
        
        # 3. Final DSA Score
        # Mean of diagonal elements
        mean_diag = torch.diag(prob_matrix).mean().item()
        
        # Random baseline is 1/K
        random_baseline = 1.0 / K
        
        # Normalized Score
        dsa_score = (mean_diag - random_baseline) / (1.0 - random_baseline)
        
        return max(0.0, dsa_score)
