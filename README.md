# Sentiment Analysis in Spanish

## Overview
This project explores sentiment analysis on Spanish-language text using multiple transformer-based models. The repository contains a comparative study of different architectures trained on various Spanish datasets, including tweets, movie reviews, and general text corpora.

## Project Structure
This repository is organized into **branches**, where each branch represents a different model implementation:

- **`bert-branch`**: Fine-tuned BERT model for Spanish sentiment analysis
- **`xml-roberta-large-branch`**: Implementation using XLM-RoBERTa-Large
- **`mdeberta-branch`**: mDeBERTa model fine-tuned for sentiment classification

Each branch contains:
- Model-specific training and evaluation scripts
- Detailed README with setup and usage instructions
- Model configuration and hyperparameters

## Datasets
The models are trained and evaluated on multiple Spanish sentiment datasets:
- **FilmAffinity Reviews**: Spanish movie reviews with sentiment labels
- **Spanish Tweets Corpus**: Annotated tweets with sentiment categories
- **Sentiment Analysis Corpus**: Additional tweet data from [bitacora-diarIA](https://github.com/Kevin-Palacios/bitacora-diarIA/blob/main/Analisis%20de%20Sentimiento/Obtencion%20de%20datos/sentiment_dataframe.csv)

## Getting Started
1. **Choose a model**: Navigate to the desired branch (BERT, XLM-RoBERTa, or mDeBERTa)
2. **Follow branch instructions**: Each branch has its own README with specific setup and usage details
3. **Review results**: Return to the main branch to see comparative analysis of every model

## Comparative Analysis
The **main branch** contains an overall analysis comparing the performance of all models across datasets. Results include:
- Accuracy, precision, recall, and F1-scores for each model
- Cross-dataset performance comparisons
- Confusion matrices and error analysis
- Visualization of results across different text domains

All analysis outputs are available in the **`analysis/`** folder, including:
- Performance comparison plots
- Dataset-specific insights
- Misclassification analysis

## Key Findings
*(TBD)*

## Requirements
Each branch has its own requirements, but common dependencies include:
- torch
- transformers
- pandas
- numpy
- matplotlib
- scikit-learn

See individual branch READMEs for specific version requirements.

## Repository Navigation
```
main/               # Overall analysis and comparisons (you are here)
├── analysis/       # Comparative visualizations and results
bert-branch/        # BERT implementation
xml-roberta-large/  # XLM-RoBERTa implementation
mdeberta-branch/    # mDeBERTa implementation
```