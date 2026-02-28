# DIAL Figure 1 Assets

This folder contains all the raw data and visual assets required to render the Overview Figure (Figure 1) for the paper.

## 📁 Directory Structure
- **frames/**: Representative PNG frames for 4 cases (Static Trap, Identity Amnesia, Context Bleeding, Ideal).
- **masks/**: Sample mask image showing Subject/Background decoupling.
- **data/**:
    - **prompts.json**: Sequential prompts used for track_s_0001.
    - **aggregated_results.json**: Final scores for all 9 SOTA models (for Scatter Plot).
    - **sample_metrics.json**: Similarity matrix M and DSA matrix P for track_s_0001.
    - **time_series_example.json**: Frame-wise metrics for sparkline plots.
    - **legacy_metrics.json**: Scores from traditional metrics to show the failure loophole.

## 🚀 How to use
You can use these JSON files with `matplotlib` or `seaborn` to generate publication-quality figures.
The matrices in `sample_metrics.json` are ready for `sns.heatmap()` rendering.
