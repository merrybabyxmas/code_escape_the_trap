#!/bin/bash
echo "Starting PhD-level full experimental pipeline..."
echo "1. Environment Setup"

# Setup FreeNoise Env
echo "Setting up FreeNoise..."
conda create -y -n freenoise python=3.8.5
conda run -n freenoise pip install -r baselines/FreeNoise/requirements.txt
# Placeholder for weight download (VideoCrafter)
# wget https://huggingface.co/VideoCrafter/Text2Video-256-v1/resolve/main/model.ckpt -O baselines/FreeNoise/checkpoints/base_256_v1/model.ckpt

# Setup StoryDiffusion Env
echo "Setting up StoryDiffusion..."
conda create -y -n storydiffusion python=3.10
conda run -n storydiffusion pip install -r baselines/StoryDiffusion/requirements.txt

# Setup DirecT2V Env
echo "Setting up DirecT2V..."
conda create -y -n direct2v python=3.9
conda run -n direct2v pip install -r baselines/DirecT2V/requirements.txt

echo "2. Running Baselines (Official Implementations)"
# Run the modified 2_run_baselines.py which uses the isolated conda environments
conda run -n paper_env python 2_run_baselines.py

echo "3. Evaluating Metrics"
conda run -n paper_env python 3_evaluate_metrics.py

echo "4. Checking Alignment and Generating Plots"
conda run -n paper_env python 4_check_alignment.py
conda run -n paper_env python 7_generate_plots.py

echo "✅ All experiments and analysis completed."
