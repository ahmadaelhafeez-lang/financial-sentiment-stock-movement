# Financial Sentiment Stock Movement Prediction

This repository contains the source code for the manuscript:

**An Explainable Time-Aware Machine Learning Evaluation of Sentiment-Enhanced Stock Movement Prediction Using Social Media Signals and Financial Text**

## Overview

This project evaluates whether sentiment features extracted from stock-related tweets and financial text provide incremental predictive value for next-day stock movement prediction.

The framework integrates:

- VADER sentiment analysis
- TextBlob polarity and subjectivity
- FinBERT financial sentiment
- Technical indicators
- Strong-movement target construction using a ±0.5% next-day return threshold
- Time-aware validation
- XGBoost, Logistic Regression, Random Forest, and Support Vector Classifier
- SHAP explainability

## Dataset Summary

The experimental dataset includes:

- 80,793 stock-related tweets
- 6,300 historical stock-price records
- 4,840 Financial PhraseBank samples
- 4,720 final stock-day observations after preprocessing and target filtering
- 25 stock tickers
- Five chronological validation folds

## Main Result

The best-performing configuration was the full-feature XGBoost model:

- F1-score: 0.513 ± 0.053
- Accuracy: 0.539 ± 0.093
- AUC: 0.519 ± 0.112
- MCC: 0.066 ± 0.187

Compared with the technical-only XGBoost baseline, the improvement was marginal. SHAP analysis showed that market-return, lagged market-return, volume-change, and momentum variables dominated model predictions, while VADER and FinBERT sentiment features acted as secondary auxiliary signals.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
