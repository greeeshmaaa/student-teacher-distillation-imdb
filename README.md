# student-teacher-distillation-imdb : IMDb Sentiment Classification via Knowledge Distillation
This project demonstrates how to distill knowledge from a powerful pretrained transformer (DistilBERT) into a lightweight CNN-based student model for sentiment classification on the IMDb dataset. The goal assigned here was to retain the teacher model's performance while significantly reducing model size and inference complexity.


# Tech Stack
- Teacher Model: `distilbert-base-uncased` (HuggingFace Transformers)
- Student Model: Lightweight CNN with GloVe-style embedding
- Libraries: PyTorch, HuggingFace Transformers, Datasets, Scikit-learn


# How to Run This Notebook (Google Colab Recommended)
> This notebook is designed to run end-to-end in Google Colab with Free GPU.

# Step-by-step Execution:
1. Run Cell 1: It will install all necessary packages (NumPy, PyTorch, Transformers)
2. After Cell 1 finishes, go to `Runtime → Restart Runtime` manually.
   - This restart is **required** to fix potential `numpy.strings` issues with Transformers.
   - Without this restart, some internal modules (like tokenizers and trainers) may not import correctly due to memory cache issues.
3. After restart, just click “Runtime → Run all” and everything will execute from start to end!


#Performance Summary

| Model                   | Accuracy | F1 Score |
|-------------------------|----------|----------|
|    Teacher (DistilBERT) | 91.19%   | 91.25%   |
|    Student (CNN)        | 87.51%   | 87.06%   |


#Compliance with Assignment Guidelines
- Used IMDb dataset
- Used pretrained transformer as teacher
- Built a smaller CNN student
- Used distillation loss (CrossEntropy + KLDiv)
- No hyperparameter tuning or excessive training
- Fully runnable on free-tier Colab (CPU/GPU)
