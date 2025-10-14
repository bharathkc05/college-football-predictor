# ============================================================
# Data Preprocessing Pipeline - No Data Leakage, Balanced Dataset
# This script takes Main.py's preprocessing approach and creates
# a clean training dataset for both sklearn and PyTorch models
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("=" * 80)
print("PREPROCESSING PIPELINE - Creating Clean Training Dataset")
print("Fix #1: Using PREVIOUS season stats (no data leakage)")
print("Fix #2: Balanced classification (equal positive/negative examples)")
print("Fix #3: Proper temporal splits with feature names preserved")
print("=" * 80)

# ---------------------------
# 1. Load CSVs
# ---------------------------
print("\n📂 Loading raw data files...")
offense = pd.read_csv('dataset/raw/all_offense_data_2015-2023_cleaned.csv')
defense = pd.read_csv('dataset/raw/all_defense_data_2015-2023_cleaned.csv')
games = pd.read_csv('dataset/raw/all_game_results_2015-2023_cleaned.csv')

print(f"✅ Offense data: {offense.shape}")
print(f"✅ Defense data: {defense.shape}")
print(f"✅ Games data: {games.shape}")

# Strip team names
offense['School'] = offense['School'].str.strip()
defense['School'] = defense['School'].str.strip()
games['Winner'] = games['Winner'].str.strip()
games['Loser'] = games['Loser'].str.strip()

# ---------------------------
# 2. Merge offense and defense features
# ---------------------------
print("\n🔄 Merging offense and defense statistics...")
teams = offense.merge(defense, on=['School','Year'], suffixes=('_off','_def'))
print(f"✅ Merged team stats: {teams.shape}")

# Get feature column names (excluding School and Year)
feature_columns = [col for col in teams.columns if col not in ['School', 'Year']]
print(f"📊 Number of features per team: {len(feature_columns)}")

# ---------------------------
# 3. Prepare BALANCED game-level features with DIFFERENTIAL computation
# ---------------------------
print("\n⚙️  Creating balanced game-level dataset...")
game_rows = []

for idx, row in games.iterrows():
    year = row['Year']
    winner = row['Winner']
    loser = row['Loser']
    winner_pts = row['Winner_Pts']
    loser_pts = row['Loser_Pts']

    # FIX #1: Use PREVIOUS year's stats to avoid data leakage
    prev_year = year - 1
    
    feat_winner = teams[(teams.School==winner) & (teams.Year==prev_year)]
    feat_loser = teams[(teams.School==loser) & (teams.Year==prev_year)]

    # Skip if missing
    if feat_winner.empty or feat_loser.empty:
        continue

    # Extract features
    X_winner = feat_winner[feature_columns].values.flatten()
    X_loser = feat_loser[feature_columns].values.flatten()

    # Compute differential features (winner - loser)
    X_diff = X_winner - X_loser

    # FIX #2: Create TWO examples per game for balanced classification
    
    # Example 1: Winner vs Loser (positive differential)
    game_dict_1 = {}
    # Differential features
    for i, col in enumerate(feature_columns):
        game_dict_1[f'{col}_diff'] = X_diff[i]
    # Raw features for score regression
    for i, col in enumerate(feature_columns):
        game_dict_1[f'team_a_{col}'] = X_winner[i]
        game_dict_1[f'team_b_{col}'] = X_loser[i]
    # Targets and metadata
    game_dict_1['final_score_diff'] = winner_pts - loser_pts
    game_dict_1['team_a_score'] = winner_pts
    game_dict_1['team_b_score'] = loser_pts
    game_dict_1['team_a'] = winner
    game_dict_1['team_b'] = loser
    game_dict_1['team_a_pts'] = winner_pts
    game_dict_1['team_b_pts'] = loser_pts
    game_dict_1['Year'] = year
    game_dict_1['label'] = 1  # team_a (winner) won
    game_rows.append(game_dict_1)
    
    # Example 2: Loser vs Winner (negative differential)
    game_dict_2 = {}
    # Differential features (reversed)
    for i, col in enumerate(feature_columns):
        game_dict_2[f'{col}_diff'] = -X_diff[i]  # Reverse the differential
    # Raw features for score regression
    for i, col in enumerate(feature_columns):
        game_dict_2[f'team_a_{col}'] = X_loser[i]
        game_dict_2[f'team_b_{col}'] = X_winner[i]
    # Targets and metadata
    game_dict_2['final_score_diff'] = loser_pts - winner_pts
    game_dict_2['team_a_score'] = loser_pts
    game_dict_2['team_b_score'] = winner_pts
    game_dict_2['team_a'] = loser
    game_dict_2['team_b'] = winner
    game_dict_2['team_a_pts'] = loser_pts
    game_dict_2['team_b_pts'] = winner_pts
    game_dict_2['Year'] = year
    game_dict_2['label'] = 0  # team_a (loser) lost
    game_rows.append(game_dict_2)

games_df = pd.DataFrame(game_rows)

print(f"✅ Original games: {len(games)}")
print(f"✅ Balanced examples created: {len(games_df)}")
print(f"✅ Class balance: {games_df['label'].value_counts().to_dict()}")

# ---------------------------
# 4. Split into train/validation/test with temporal ordering
# ---------------------------
print("\n📊 Creating temporal splits...")

# Training: 2016-2021 (60%)
# Validation: 2022 (20%)
# Test: 2023 (20%)
train_df = games_df[games_df.Year <= 2021]
val_df = games_df[games_df.Year == 2022]
test_df = games_df[games_df.Year == 2023]

print(f"✅ Train set (2016-2021): {len(train_df)} examples")
print(f"✅ Validation set (2022): {len(val_df)} examples")
print(f"✅ Test set (2023): {len(test_df)} examples")

# ---------------------------
# 5. Extract features for both approaches
# ---------------------------
print("\n🔧 Preparing feature datasets...")

# Get differential feature columns
diff_columns = [col for col in games_df.columns if col.endswith('_diff') and col != 'final_score_diff']
# Get raw feature columns for score regression
team_a_columns = [col for col in games_df.columns if col.startswith('team_a_') and col not in ['team_a', 'team_a_pts', 'team_a_score']]
team_b_columns = [col for col in games_df.columns if col.startswith('team_b_') and col not in ['team_b', 'team_b_pts', 'team_b_score']]
raw_columns = team_a_columns + team_b_columns

metadata_columns = ['team_a', 'team_b', 'team_a_pts', 'team_b_pts', 'Year', 'label']
target_diff = 'final_score_diff'
target_score_a = 'team_a_score'
target_score_b = 'team_b_score'

print(f"📊 Differential features: {len(diff_columns)}")
print(f"📊 Raw features (team_a): {len(team_a_columns)}")
print(f"📊 Raw features (team_b): {len(team_b_columns)}")
print(f"📊 Total raw features: {len(raw_columns)}")
print(f"📋 Metadata columns: {metadata_columns}")
print(f"🎯 Target columns: {target_diff}, {target_score_a}, {target_score_b}")

# Create differential feature datasets (for differential regression)
train_features = train_df[diff_columns + [target_diff]]
val_features = val_df[diff_columns + [target_diff]]
test_features = test_df[diff_columns + [target_diff]]

# Create raw feature datasets (for score regression)
train_features_raw = train_df[raw_columns + [target_score_a, target_score_b]]
val_features_raw = val_df[raw_columns + [target_score_a, target_score_b]]
test_features_raw = test_df[raw_columns + [target_score_a, target_score_b]]

# Create full datasets with metadata (for analysis)
train_full = train_df[diff_columns + [target_diff, target_score_a, target_score_b] + metadata_columns]
val_full = val_df[diff_columns + [target_diff, target_score_a, target_score_b] + metadata_columns]
test_full = test_df[diff_columns + [target_diff, target_score_a, target_score_b] + metadata_columns]

# ---------------------------
# 6. Standardize features
# ---------------------------
print("\n📏 Standardizing features...")

# Fit scalers on training data only
scaler_diff = StandardScaler()
scaler_diff.fit(train_features[diff_columns])

scaler_raw = StandardScaler()
scaler_raw.fit(train_features_raw[raw_columns])

# Transform differential features
train_features_scaled = train_features.copy()
val_features_scaled = val_features.copy()
test_features_scaled = test_features.copy()

train_features_scaled[diff_columns] = scaler_diff.transform(train_features[diff_columns])
val_features_scaled[diff_columns] = scaler_diff.transform(val_features[diff_columns])
test_features_scaled[diff_columns] = scaler_diff.transform(test_features[diff_columns])

# Transform raw features
train_features_raw_scaled = train_features_raw.copy()
val_features_raw_scaled = val_features_raw.copy()
test_features_raw_scaled = test_features_raw.copy()

train_features_raw_scaled[raw_columns] = scaler_raw.transform(train_features_raw[raw_columns])
val_features_raw_scaled[raw_columns] = scaler_raw.transform(val_features_raw[raw_columns])
test_features_raw_scaled[raw_columns] = scaler_raw.transform(test_features_raw[raw_columns])

print("✅ Differential features standardized (mean=0, std=1)")
print("✅ Raw features standardized (mean=0, std=1)")

# ---------------------------
# 7. Save datasets (ONLY files used by advanced_models.py)
# ---------------------------
print("\n💾 Saving datasets...")

# Create directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)
os.makedirs('dataset/processed', exist_ok=True)

# Save scaled differential features (USED)
train_features_scaled.to_csv('dataset/processed/train_features_scaled.csv', index=False)
val_features_scaled.to_csv('dataset/processed/val_features_scaled.csv', index=False)
test_features_scaled.to_csv('dataset/processed/test_features_scaled.csv', index=False)
print(f"✅ Saved scaled differential features: train/val/test_features_scaled.csv")

# Save scaled raw features for score regression (USED)
train_features_raw_scaled.to_csv('dataset/processed/train_features_raw_scaled.csv', index=False)
val_features_raw_scaled.to_csv('dataset/processed/val_features_raw_scaled.csv', index=False)
test_features_raw_scaled.to_csv('dataset/processed/test_features_raw_scaled.csv', index=False)
print(f"✅ Saved scaled raw features: train/val/test_features_raw_scaled.csv")

# Save full datasets with metadata (USED)
train_full.to_csv('dataset/processed/train_full.csv', index=False)
val_full.to_csv('dataset/processed/val_full.csv', index=False)
test_full.to_csv('dataset/processed/test_full.csv', index=False)
print(f"✅ Saved full datasets with metadata: train/val/test_full.csv")

# ---------------------------
# 8. Summary Statistics
# ---------------------------
print("\n" + "=" * 80)
print("📊 DATASET SUMMARY")
print("=" * 80)

print(f"\n🎯 Target Variable Statistics:")
print(f"   Differential Target ({target_diff}):")
print(f"      Train - Mean: {train_features[target_diff].mean():.2f}, Std: {train_features[target_diff].std():.2f}")
print(f"      Val   - Mean: {val_features[target_diff].mean():.2f}, Std: {val_features[target_diff].std():.2f}")
print(f"      Test  - Mean: {test_features[target_diff].mean():.2f}, Std: {test_features[target_diff].std():.2f}")
print(f"   Score Targets ({target_score_a}, {target_score_b}):")
print(f"      Train - team_a Mean: {train_features_raw[target_score_a].mean():.2f}, team_b Mean: {train_features_raw[target_score_b].mean():.2f}")
print(f"      Val   - team_a Mean: {val_features_raw[target_score_a].mean():.2f}, team_b Mean: {val_features_raw[target_score_b].mean():.2f}")
print(f"      Test  - team_a Mean: {test_features_raw[target_score_a].mean():.2f}, team_b Mean: {test_features_raw[target_score_b].mean():.2f}")

print(f"\n📊 Class Balance (Winner Prediction):")
print(f"   Train - {train_full['label'].value_counts().to_dict()}")
print(f"   Val   - {val_full['label'].value_counts().to_dict()}")
print(f"   Test  - {test_full['label'].value_counts().to_dict()}")

print(f"\n📅 Temporal Distribution:")
print(f"   Train: Years {train_full['Year'].min()}-{train_full['Year'].max()}")
print(f"   Val:   Year {val_full['Year'].min()}")
print(f"   Test:  Year {test_full['Year'].min()}")

print(f"\n📈 Feature Statistics:")
print(f"   Differential features: {len(diff_columns)}")
print(f"   Raw features (total): {len(raw_columns)} ({len(team_a_columns)} per team)")
print(f"   Sample differential features: {diff_columns[:5]}")
print(f"   Sample raw features: {team_a_columns[:3]}")

print("\n" + "=" * 80)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 80)
print("\n📝 Files generated (only what's needed for advanced_models.py):")
print("   🔵 Scaled Differential Features (for differential regression):")
print("      - dataset/processed/train_features_scaled.csv")
print("      - dataset/processed/val_features_scaled.csv")
print("      - dataset/processed/test_features_scaled.csv")
print("   🟢 Scaled Raw Features (for score regression):")
print("      - dataset/processed/train_features_raw_scaled.csv")
print("      - dataset/processed/val_features_raw_scaled.csv")
print("      - dataset/processed/test_features_raw_scaled.csv")
print("   📊 Full Datasets (with metadata for analysis):")
print("      - dataset/processed/train_full.csv")
print("      - dataset/processed/val_full.csv")
print("      - dataset/processed/test_full.csv")
print("=" * 80)
