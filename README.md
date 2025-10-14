# Advanced College Football Prediction Models

A comprehensive machine learning analysis comparing **Score Regression** and **Differential Regression** approaches for predicting college football game outcomes (2016-2023).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Data Pipeline](#data-pipeline)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)
- [Requirements](#requirements)

---

## 🎯 Overview

This project implements and compares **9 different machine learning models** to predict college football game winners using historical team statistics. The models are evaluated on their ability to:
- Predict individual team scores (Score Regression)
- Predict score differentials directly (Differential Regression)
- Classify game winners (Binary Classification)

### Key Features
✅ **No Data Leakage**: Uses previous season's statistics to predict current season outcomes  
✅ **Balanced Dataset**: Equal positive and negative examples for fair evaluation  
✅ **Temporal Splits**: Proper train/validation/test splits maintaining chronological order  
✅ **Feature Selection**: Multiple techniques including SelectKBest, RFE, and Alpha-Selection  
✅ **Comprehensive Evaluation**: Year-by-year accuracy tracking from 2016-2023  

---

## 🚀 Getting Started

### Quick Start Guide

1. **Clone the Repository** (if from GitHub)
   ```bash
   git clone https://github.com/bharathkc05/college-football-predictor.git
   cd college-football-predictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   - Place raw CSV files in `dataset/raw/` (or run `python setup_data.py` if files are in root)

4. **Run Preprocessing**
   ```bash
   python preprocessing_pipeline.py
   ```

5. **Run Analysis**
   ```bash
   python bowl-game-predictor.py
   ```

6. **View Results**
   - Check `results/` folder for visualizations and CSV outputs

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: ~100 MB for code and processed data
- **OS**: Windows, macOS, or Linux

---

## 📁 Project Structure

```
.
├── bowl-game-predictor.py                    # Main analysis script
├── preprocessing_pipeline.py              # Data preprocessing pipeline
├── setup_data.py                          # Helper script to organize raw data files
├── bowl-game-predictor_README.md             # This file
│
├── dataset/
│   ├── raw/                               # Raw input data files
│   │   ├── all_offense_data_2015-2023_cleaned.csv
│   │   ├── all_defense_data_2015-2023_cleaned.csv
│   │   └── all_game_results_2015-2023_cleaned.csv
│   └── processed/
│       ├── train_features_scaled.csv      # Scaled differential features (train)
│       ├── val_features_scaled.csv        # Scaled differential features (validation)
│       ├── test_features_scaled.csv       # Scaled differential features (test)
│       ├── train_features_raw_scaled.csv  # Scaled raw features (train)
│       ├── val_features_raw_scaled.csv    # Scaled raw features (validation)
│       ├── test_features_raw_scaled.csv   # Scaled raw features (test)
│       ├── train_full.csv                 # Full train data with metadata
│       ├── val_full.csv                   # Full validation data with metadata
│       └── test_full.csv                  # Full test data with metadata
│
└── results/
    ├── advanced_model_comparison.csv              # Model performance metrics
    ├── year_by_year_accuracy.csv                  # Annual accuracy breakdown
    ├── bowl-game-predictor_visualization.png          # 6-panel comparison plot
    └── year_by_year_prediction_accuracy.png       # Temporal accuracy trends
```

---

## 🔬 Methodology

### Data Preprocessing

**Input Data (in `dataset/raw/`):**
- `all_offense_data_2015-2023_cleaned.csv` (1,168 team-seasons, 26 features)
- `all_defense_data_2015-2023_cleaned.csv` (1,168 team-seasons, 26 features)
- `all_game_results_2015-2023_cleaned.csv` (7,679 games)

**Processing Steps:**
1. **Merge** offense and defense statistics → 48 features per team
2. **Temporal alignment**: Use year `t-1` stats to predict year `t` outcomes (prevents data leakage)
3. **Feature engineering**: Compute differential features (Team A - Team B)
4. **Balancing**: Create 2 examples per game (A vs B, B vs A) → 9,872 balanced examples
5. **Scaling**: StandardScaler (mean=0, std=1) fitted on training data only
6. **Temporal split**:
   - **Train**: 2016-2021 (7,316 examples)
   - **Validation**: 2022 (1,268 examples)
   - **Test**: 2023 (1,288 examples)

---

## 🤖 Models Implemented

### Part 1: Score Regression (Predicting Individual Team Scores)

| Model | Description | Features | Key Characteristics |
|-------|-------------|----------|---------------------|
| **LR_Baseline** | Linear Regression | 46 per team (92 total) | Predicts team_a_score and team_b_score separately |
| **LR_SelectKBest** | Linear Regression + Feature Selection | Best K via validation | Selects top K features using F-statistic |
| **LR_Forward** | Linear Regression + Forward Stepwise | 20 per team | RFE with step=1 (adds features incrementally) |
| **LR_Backward** | Linear Regression + Backward Stepwise | 20 per team | RFE with step=5 (removes features in batches) |
| **SVR** | Support Vector Regression | 46 per team | Linear kernel, trained on 2,000 samples |
| **LogisticRegression** | Binary Classifier | 46 differential | Direct winner prediction (no score prediction) |

### Part 2: Differential Regression (Predicting Score Difference)

| Model | Description | Features | Key Characteristics |
|-------|-------------|----------|---------------------|
| **LR_Diff_Baseline** | Linear Regression on Differential | 46 | Predicts score_diff = score_a - score_b |
| **LR_Diff_SelectKBest** | LR + Feature Selection | Best K via validation | F-statistic based selection |
| **LR_Diff_AlphaSelection** | LR + Stability-based Selection | 20 | Features appearing in ≥20% of 50 randomizations |

---

## 📊 Results

### Overall Performance (Test Set - 2023 Season)

| Rank | Model | Type | Accuracy | F1 Score | MAE (Diff) |
|------|-------|------|----------|----------|------------|
| 🥇 1 | **LR_Diff_SelectKBest** | Differential | **66.5%** | **0.665** | 14.73 pts |
| 🥈 2 | **LR_Diff_Baseline** | Differential | **66.5%** | **0.665** | 14.73 pts |
| 🥉 3 | **LogisticRegression** | Score (Classifier) | **65.8%** | **0.658** | - |
| 4 | LR_Diff_AlphaSelection | Differential | 64.9% | 0.649 | 14.67 pts |
| 5 | LR_Baseline | Score | 62.6% | 0.626 | - |
| 6 | LR_SelectKBest | Score | 62.6% | 0.626 | - |
| 7 | LR_Backward | Score | 62.1% | 0.621 | - |
| 8 | LR_Forward | Score | 60.4% | 0.604 | - |
| 9 | SVR | Score | 55.7% | 0.575 | - |

### Year-by-Year Accuracy (2016-2023)

**Best Model (LR_Diff_SelectKBest):**

| Year | Accuracy | Games | Correct | Incorrect |
|------|----------|-------|---------|-----------|
| 2016 | 65.9% | ~1,235 | 814 | 421 |
| 2017 | 60.7% | ~1,234 | 749 | 485 |
| 2018 | 66.1% | ~1,212 | 801 | 411 |
| 2019 | 65.5% | ~1,245 | 815 | 430 |
| 2020 | 66.4% | ~638 | 424 | 214 |
| 2021 | 61.6% | ~1,270 | 782 | 488 |
| 2022 | 62.5% | 1,268 | 792 | 476 |
| 2023 | 66.5% | 1,288 | 856 | 432 |

**Average Accuracy Across All Years: 64.4%**

---

## 🔄 Data Pipeline

### Preprocessing Pipeline (`preprocessing_pipeline.py`)

```
Raw CSV Files (dataset/raw/)
    ↓
Merge Offense + Defense → 48 features per team
    ↓
Temporal Alignment (use year t-1 to predict year t)
    ↓
Feature Engineering (differential features)
    ↓
Balance Dataset (2 examples per game)
    ↓
Temporal Split (Train/Val/Test)
    ↓
Feature Scaling (StandardScaler)
    ↓
Save Processed Datasets (dataset/processed/)
```

**Run preprocessing:**
```bash
python preprocessing_pipeline.py
```

**Output:**
- 9 CSV files in `dataset/processed/`
- Summary statistics printed to console
- ~30 seconds execution time

---

## 📈 Visualizations

### 1. Advanced Models Visualization (`bowl-game-predictor_visualization.png`)

**6-panel figure showing:**

| Panel | Content | Insights |
|-------|---------|----------|
| Top Left | Model Accuracy Comparison | Bar chart of all 9 models ranked by accuracy |
| Top Middle | F1 Score Comparison | Performance on balanced dataset |
| Top Right | Probability Distribution | Histogram showing LogisticRegression calibration |
| Bottom Left | Differential Regression Scatter | Predicted vs Actual score differential |
| Bottom Middle | Confusion Matrix (Best Score) | LogisticRegression error breakdown |
| Bottom Right | Confusion Matrix (Best Diff) | LR_Diff_SelectKBest error breakdown |

### 2. Year-by-Year Accuracy (`year_by_year_prediction_accuracy.png`)

**2-panel temporal analysis:**
- Left: Score Regression models (6 lines)
- Right: Differential Regression models (3 lines)
- Shows stability and trends from 2016-2023

---

## 🚀 How to Run

### Prerequisites

#### Install Required Packages

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual Installation**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Required Dependencies:**
- `pandas >= 1.3.0` - Data manipulation and analysis
- `numpy >= 1.20.0` - Numerical computing
- `scikit-learn >= 0.24.0` - Machine learning algorithms
- `matplotlib >= 3.3.0` - Plotting and visualization
- `seaborn >= 0.11.0` - Statistical data visualization


### Step 1: Preprocess Data
```bash
python preprocessing_pipeline.py
```

**Expected output:**
```
✅ Saved scaled differential features: train/val/test_features_scaled.csv
✅ Saved scaled raw features: train/val/test_features_raw_scaled.csv
✅ Saved full datasets with metadata: train/val/test_full.csv
```

### Step 2: Run Analysis
```bash
python bowl-game-predictor.py
```

**Expected output:**
```
================================================================================
ADVANCED COLLEGE FOOTBALL PREDICTION MODELS
================================================================================
...
[BEST] BEST OVERALL MODEL: LR_Diff_SelectKBest
   Accuracy: 0.665 (66.5%)
   F1 Score: 0.665
================================================================================
```

**Execution time:** ~2-3 minutes

### Step 3: View Results

**CSV Files:**
- `results/advanced_model_comparison.csv` - Model metrics table
- `results/year_by_year_accuracy.csv` - Annual breakdown

**Visualizations:**
- `results/bowl-game-predictor_visualization.png` - Main comparison figure
- `results/year_by_year_prediction_accuracy.png` - Temporal trends

---

## 🔍 Key Findings

### 1. **Differential Regression Outperforms Score Regression**
- **Best Differential Model**: 66.5% accuracy (LR_Diff_SelectKBest)
- **Best Score Model**: 65.8% accuracy (LogisticRegression)
- **Reason**: Predicting score difference directly is more efficient than predicting two separate scores

### 2. **Feature Selection Maintains Performance**
- SelectKBest (46 features) = Baseline (46 features) in accuracy
- Suggests many features are redundant or provide minimal information
- **Benefit**: Simpler models with same performance

### 3. **Improvement Over Random Baseline**
- Random guessing: 50.0%
- Best model: 66.5%
- **Absolute improvement**: +16.5 percentage points
- **Relative improvement**: 33% better than random

### 4. **Temporal Stability**
- Models maintain 60-67% accuracy across all years (2016-2023)
- No significant performance degradation over time
- 2020 shows slightly higher variance (COVID-affected season with fewer games)

### 5. **Linear Models Competitive with Non-linear**
- Support Vector Regression (SVR) performs worse than Linear Regression
- **Reason**: Either the relationship is approximately linear, or SVR needs more tuning/data

### 6. **Logistic Regression Shows Good Calibration**
- Probability distribution histogram shows clear separation
- High-confidence predictions (>0.7 or <0.3) are generally correct
- Well-calibrated probabilities useful for betting/decision-making

---

## 📦 Requirements

### Core Dependencies
```
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

### Optional (for GPU acceleration - not used in this version)
```
torch >= 1.9.0
```

### Python Version
- **Required**: Python 3.8+
- **Tested on**: Python 3.12

---

## 📝 Notes

### Data Leakage Prevention
- ✅ Uses year `t-1` statistics to predict year `t` outcomes
- ✅ No future information leaked into training
- ✅ Scaler fitted only on training data

### Class Balance
- ✅ Perfectly balanced dataset (50% wins, 50% losses)
- ✅ Accuracy = F1 Score (expected for balanced data)
- ✅ No need for class weights or oversampling

### Feature Exclusions
- ❌ `Pts_off` (points scored offense) - direct target leakage
- ❌ `Pts_def` (points allowed defense) - direct target leakage
- ✅ All other offensive and defensive statistics included

---

## 🎓 Model Interpretation

### What the Models Predict:

**Score Regression Models:**
- Predict: `team_a_score` and `team_b_score`
- Winner: Team with higher predicted score
- Use case: When you need point spread predictions

**Differential Regression Models:**
- Predict: `score_diff = team_a_score - team_b_score`
- Winner: If `score_diff > 0`, team_a wins
- Use case: When you only care about winner, not exact scores

**Logistic Regression:**
- Predict: `P(team_a wins | features)`
- Winner: If `P > 0.5`, team_a wins
- Use case: When you want confidence/probability estimates

---

## 🏆 Best Model Recommendation

**For Winner Prediction:** `LR_Diff_SelectKBest` or `LR_Diff_Baseline`
- **Accuracy**: 66.5%
- **Simplicity**: Linear model, easy to interpret
- **Speed**: Fast training and prediction
- **Stability**: Consistent across years

**For Probability Estimates:** `LogisticRegression`
- **Accuracy**: 65.8%
- **Calibration**: Well-calibrated probabilities
- **Use case**: Betting, risk assessment

**For Score Predictions:** `LR_Baseline` or `LR_SelectKBest`
- **Accuracy**: 62.6%
- **MAE**: ~10.7 points per team
- **Use case**: Point spread betting, detailed analysis

---

## 📧 Contact & Attribution

**Project**: Advanced College Football Prediction Models  
**Dataset**: NCAA FBS Statistics (2015-2023)  
**License**: Educational/Research Use  

---

## 🔮 Future Improvements

### Potential Enhancements:
1. **Feature Engineering**:
   - Home/away game indicators
   - Win streak/momentum features
   - Head-to-head historical matchups
   - Strength of schedule metrics

2. **Advanced Models**:
   - Gradient Boosting (XGBoost, LightGBM)
   - Random Forests
   - Neural Networks
   - Ensemble methods

3. **Temporal Models**:
   - LSTM for sequence prediction
   - Time-series specific features
   - Seasonal trends

4. **Betting Applications**:
   - Point spread predictions
   - Over/under totals
   - Money line odds conversion

---

## 📊 Quick Stats Summary

| Metric | Value |
|--------|-------|
| **Total Models** | 9 |
| **Best Accuracy** | 66.5% |
| **Improvement over Random** | +16.5% |
| **Training Examples** | 7,316 |
| **Test Examples** | 1,288 |
| **Features (Differential)** | 46 |
| **Features (Raw per team)** | 46 |
| **Years Analyzed** | 2016-2023 (8 years) |
| **Total Games Analyzed** | 9,872 |
| **Average Score Differential** | 0.0 points (balanced) |
| **MAE (Best Model)** | 14.73 points |

---

**Last Updated**: October 2025  
**Status**: ✅ Complete and Ready for Use
