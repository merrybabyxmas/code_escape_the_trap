# Human Evaluation Protocol

Please watch the videos in the `video_samples` folder and fill out the `human_evaluation_sheet.csv`.

### Scoring Criteria (1: Worst, 5: Best)
- **Human_Subject_Consistency:** Is the main subject identical across shots without morphing?
- **Human_Background_Dynamics:** Does the background properly change (Track S) or stay stable (Track M)?
- **Human_Prompt_Adherence:** Does the video actually follow the action/camera instructions?
- **Human_Cut_Sharpness:** Is the transition a sharp cinematic cut (5) or a messy fade/morph (1)?
- **Notes_or_Amnesia_Observed:** Mark 'O' if the subject's identity completely breaks or fuses with background.

After completion, we will use this CSV to compute Pearson/Spearman correlation with our DSA and Decoupled metrics to validate human alignment.
