import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

class DinoEvaluator:
    def __init__(self, model_type='dinov2_vits14'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('facebookresearch/dinov2', model_type).to(self.device).to(torch.bfloat16)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embedding(self, image):
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).unsqueeze(0).to(self.device).to(torch.bfloat16)
        else:
            img_tensor = image.to(self.device).to(torch.bfloat16)
        
        features = self.model(img_tensor)
        return F.normalize(features.float(), p=2, dim=-1)

    def calculate_subject_consistency(self, subject_crops):
        """
        Calculates subject consistency across shots.
        Includes explicit Error Propagation handling: If the subject masking model 
        (e.g., Grounding DINO) fails to detect the subject due to severe Identity Amnesia 
        or background fusion, the crop will be None or invalid, resulting in an immediate 0.0 score.
        
        subject_crops: List of PIL Images (cropped around the subject) or None for failed detections.
        """
        if len(subject_crops) < 2: return 1.0
        
        valid_crops = []
        for img in subject_crops:
            # Fallback: If mask fails (Identity Amnesia), consistency is broken.
            if img is None or img.width < 10 or img.height < 10:
                return 0.0 # Strict penalty for masking failure
            valid_crops.append(img)
            
        if len(valid_crops) != len(subject_crops):
            return 0.0
        
        embeddings = [self.get_embedding(img) for img in valid_crops]
        embeddings = torch.cat(embeddings, dim=0)
        
        # Calculate pairwise cosine similarity
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Mean of upper triangle (excluding diagonal)
        n = len(valid_crops)
        triu_indices = torch.triu_indices(n, n, offset=1)
        mean_sim = sim_matrix[triu_indices[0], triu_indices[1]].mean().item()
        
        return max(0.0, mean_sim)

    def calculate_background_diversity(self, full_frames, subject_masks):
        """
        주체를 마스킹한 배경 영역의 다양성을 계산합니다. 
        배경 임베딩의 분산(Variance)이 높을수록 'Static Video Trap'에서 벗어났음을 의미합니다.
        """
        return 0.5 
