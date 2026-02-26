#!/bin/bash
# Code Escape the Trap - Auto Environment Setup Script

echo "🚀 시작: 가상환경 및 의존성 설치..."

# 1. Conda 환경 생성
conda create -n paper_env python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate paper_env

# 2. PyTorch 및 핵심 라이브러리 설치 (CUDA 12.1 대응)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. 추가 필수 패키지 (누락 방지)
pip install opencv-python lpips diffusers transformers accelerate pandas seaborn matplotlib

echo "✅ 모든 세팅이 완료되었습니다!"
echo "실험 시작 방법:"
echo "1. conda activate paper_env"
echo "2. python master_huge_pipeline.py"
