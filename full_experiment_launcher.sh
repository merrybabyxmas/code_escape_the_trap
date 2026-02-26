#!/bin/bash
PROJECT_ROOT="/home/dongwoo43/paper_escapethetrap/escapethetrap"
cd $PROJECT_ROOT

export PYTHONUNBUFFERED=1

MODELS=("CogVideoX" "LTX-Video" "SVD" "StoryDiffusion" "FreeNoise" "ModelScope" "AnimateDiff" "Mora" "DirecT2V")

echo "Starting REAL-TIME INCREMENTAL FULL PhD-level Experiment..."
echo "Results will be saved to per_sample_results_full.json after EACH video."

for model in "${MODELS[@]}"
do
    echo "--------------------------------------------------"
    echo "🚀 Processing Model: $model"
    echo "--------------------------------------------------"
    
    # 1. Generate & Evaluate (Integrated)
    /home/dongwoo43/miniconda3/envs/paper_env/bin/python -u run_full_benchmark.py --model "$model"
    
    # 2. Finalize Summaries for this model
    echo "📊 Finalizing summaries for $model..."
    /home/dongwoo43/miniconda3/envs/paper_env/bin/python -u 3_evaluate_metrics.py
    
    # 3. Update Plots and Paper
    echo "📈 Updating Radar Charts and Paper Content..."
    /home/dongwoo43/miniconda3/envs/paper_env/bin/python -u 7_generate_plots.py
    /home/dongwoo43/miniconda3/envs/paper_env/bin/python -u 6_write_paper_content.py
    
    echo "✅ Model $model finished and integrated into paper."
done

echo "🎉 ALL 100 scenarios for ALL 9 models processed and paper finalized."
