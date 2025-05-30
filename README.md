# Music Track Popularity Prediction 🎵

This project focuses on predicting the popularity score (0–100) of music tracks using advanced machine learning regression models. The dataset, sourced from a Kaggle competition, includes audio features, metadata, and artist-level information. The models explored include XGBoost and CatBoost, with and without hyperparameter tuning (GridSearchCV & RandomizedSearchCV). Multiple evaluation metrics were used to assess model performance.

## 📊 Problem Overview

Music streaming services and record labels use data-driven methods to predict which tracks will perform well. This project builds a regression pipeline that learns from track features to forecast their popularity, helping to identify successful tracks in advance.

## ⚙️ Workflow

- Data Exploration and Cleaning
- Feature Encoding and Imputation
- Feature Selection (Spearman Correlation, Mutual Information)
- Feature Scaling
- Model Training:
  - Linear Regression (baseline)
  - XGBoost Regressor (tuned via GridSearchCV & RandomizedSearchCV)
  - CatBoost Regressor (tuned via GridSearchCV)
- 5-Fold Cross-Validation
- Evaluation using RMSE, MAE, R², MAPE, SMAPE, and MedAE

## 📈 Results Summary

The best-performing model was **XGBoost with RandomizedSearchCV**, which achieved:
- RMSE: 8.03
- MAE: 4.07
- R² Score: 0.861
- SMAPE: 13.52


## 📌 Key Technologies

- Python 3
- XGBoost
- CatBoost
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-popularity-prediction.git
   cd music-popularity-prediction
