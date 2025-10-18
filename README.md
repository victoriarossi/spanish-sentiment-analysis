# Sentiment Analysis in Spanish

## Overview
This project fine-tunes and evaluates a mDeBERTa transformer model for sentiment analysis on Spanish-language datasets. It compares model performance across multiple datasets to understand how well each captures sentiment nuances in different text domains.

## Objectives
- Fine-tune a pretrained Spanish mDeBERTa model on annotated datasets.  
- Evaluate model accuracy, precision, recall, and F1-score.  
- Analyze misclassifications and label distribution across datasets.  
- Compare results to identify dataset-specific challenges.

## Datasets
The project uses multiple annotated datasets:
- **FilmAffinity Reviews:** Spanish movie reviews labeled by sentiment.  
- **Spanish Tweets Corpus:** Tweets annotated with sentiment categories.  
- **Custom Dataset:** Manually annotated texts for validation.

Each dataset includes:
content,pred_label,pred_id,pred_confidence,gold_label


## Model
- **Base model:** `microsoft/mdeberta-v3-base`  
- **Framework:** PyTorch with HuggingFace Transformers  
- **Tokenizer:** DebertaV2Tokenizer  
- **Optimizer:** AdamW  
- **Train/Test split:** 80/20  

## Pipeline
1. **Preprocessing:** Clean text (normalize, remove symbols, lowercase).  
2. **Tokenization:** Convert sentences to input IDs and attention masks.  
3. **Training:** Fine-tune mDeBERTa on sentiment labels using cross-entropy loss.  
4. **Evaluation:** Generate predictions and compute metrics.  
5. **Analysis:** Visualize results and compare performance across datasets.

## Usage
Run training:
```bash
python3 training_model.py
```
Run predictions:
```bash
python3 test_polarity.py
```
Run analysis:
```bash
python3 analyzing_predictions.py
```

## Output includes:
- Dataset sizes
- Training progress logs
- Accuracy and F1 scores
- Confusion matrices and comparison plots

## Requirements
- torch
- transformers
- pandas
- numpy
- matplotlib
- scikit-learn

