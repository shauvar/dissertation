"""
Robust Model Validation with Cross-Validation
Addresses overfitting concerns for small datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("=" * 70)
print("ROBUST MODEL VALIDATION - Cross-Validation Analysis")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================

PLATFORM = 'instagram'  # Change to 'instagram' for Instagram analysis

if PLATFORM == 'youtube':
    df = pd.read_csv('youtube_features_final.csv')
    print(f"\n Loaded YouTube data: {len(df)} channels")
else:
    df = pd.read_csv('instagram_features_final.csv')
    print(f"\n Loaded Instagram data: {len(df)} accounts")

# Target variable
target = 'engagement_rate_mean'

# Exclude non-features
exclude_cols = [
    'channel_name', 'channel_id', 'username', 'rank', 'country',
    'engagement_rate_mean', 'engagement_rate_60day'
]

feature_cols = [col for col in df.columns 
                if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

print(f"Features: {len(feature_cols)}")
print(f"Sample size: {len(df)}")
print(f"Samples per feature: {len(df)/len(feature_cols):.1f}")

# Prepare data
X = df[feature_cols]
y = df[target]

# =============================================================================
# DEFINE MODELS WITH APPROPRIATE COMPLEXITY
# =============================================================================

print("\n" + "=" * 70)
print("MODEL DEFINITIONS")
print("=" * 70)

# Simple models (appropriate for small datasets)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10)': Ridge(alpha=10.0),
    'Lasso (α=0.01)': Lasso(alpha=0.01, max_iter=10000),
    'Random Forest (Simple)': RandomForestRegressor(
        n_estimators=50,
        max_depth=3,
        min_samples_split=5,
        random_state=42
    ),
    'Random Forest (Moderate)': RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    ),
    'XGBoost (Simple)': XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    ),
}

print(f"\n Testing {len(models)} models with varying complexity")

# =============================================================================
# CROSS-VALIDATION SETUP
# =============================================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION SETUP")
print("=" * 70)

# Use 5-fold CV (appropriate for sample size 40-45)
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\n Using {n_folds}-Fold Cross-Validation")
print(f"   Each fold: ~{len(df)//n_folds} samples")
print(f"   Training per fold: ~{len(df)*(n_folds-1)//n_folds} samples")
print(f"   Testing per fold: ~{len(df)//n_folds} samples")

# Define scoring metrics
scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

# =============================================================================
# RUN CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("RUNNING CROSS-VALIDATION")
print("=" * 70)

results = []

for name, model in models.items():
    print(f"\n Training: {name}...")
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate metrics
    train_r2_mean = cv_results['train_r2'].mean()
    train_r2_std = cv_results['train_r2'].std()
    test_r2_mean = cv_results['test_r2'].mean()
    test_r2_std = cv_results['test_r2'].std()
    
    test_rmse_mean = np.sqrt(-cv_results['test_neg_mse'].mean())
    test_mae_mean = -cv_results['test_neg_mae'].mean()
    
    # Calculate overfitting indicator
    overfitting_gap = train_r2_mean - test_r2_mean
    
    results.append({
        'Model': name,
        'Train R² (mean)': train_r2_mean,
        'Train R² (std)': train_r2_std,
        'Test R² (mean)': test_r2_mean,
        'Test R² (std)': test_r2_std,
        'Test RMSE': test_rmse_mean,
        'Test MAE': test_mae_mean,
        'Overfitting Gap': overfitting_gap
    })
    
    print(f"   Train R²: {train_r2_mean:.4f} (±{train_r2_std:.4f})")
    print(f"   Test R²:  {test_r2_mean:.4f} (±{test_r2_std:.4f})")
    print(f"   Overfitting gap: {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.20:
        print(f"     HIGH OVERFITTING RISK!")
    elif overfitting_gap > 0.10:
        print(f"     Moderate overfitting")
    else:
        print(f"    Good generalization")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R² (mean)', ascending=False)

print("\n All Models (sorted by Test R²):")
print(results_df.to_string(index=False, float_format='%.4f'))

# Save results
results_df.to_csv(f'{PLATFORM}_cv_results.csv', index=False)
print(f"\n Saved: {PLATFORM}_cv_results.csv")

# Find best model
best_model = results_df.iloc[0]
print(f"\n Best Model (by CV Test R²): {best_model['Model']}")
print(f"   Test R²: {best_model['Test R² (mean)']:.4f} (±{best_model['Test R² (std)']:.4f})")
print(f"   Test RMSE: {best_model['Test RMSE']:.4f}")
print(f"   Overfitting gap: {best_model['Overfitting Gap']:.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: Train vs Test R²
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R² comparison
x_pos = np.arange(len(results_df))
width = 0.35

axes[0].bar(x_pos - width/2, results_df['Train R² (mean)'], width, 
           label='Train R²', alpha=0.8, color='skyblue', edgecolor='black')
axes[0].bar(x_pos + width/2, results_df['Test R² (mean)'], width,
           label='Test R² (CV)', alpha=0.8, color='coral', edgecolor='black')

axes[0].set_xlabel('Model', fontsize=11)
axes[0].set_ylabel('R² Score', fontsize=11)
axes[0].set_title('Train vs Test R² - Cross-Validation', fontsize=13, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.70)')
axes[0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.60)')

# Overfitting gap
axes[1].bar(range(len(results_df)), results_df['Overfitting Gap'], 
           color='red', alpha=0.7, edgecolor='black')
axes[1].axhline(y=0.10, color='orange', linestyle='--', linewidth=2, 
               label='Moderate threshold (0.10)')
axes[1].axhline(y=0.20, color='red', linestyle='--', linewidth=2,
               label='High risk threshold (0.20)')
axes[1].set_xlabel('Model', fontsize=11)
axes[1].set_ylabel('Overfitting Gap (Train R² - Test R²)', fontsize=11)
axes[1].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(len(results_df)))
axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLATFORM}_cv_train_vs_test.png', dpi=300, bbox_inches='tight')
print(f" Saved: {PLATFORM}_cv_train_vs_test.png")
plt.close()

# Figure 2: Test R² with error bars
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(results_df))
ax.bar(x_pos, results_df['Test R² (mean)'], 
       yerr=results_df['Test R² (std)'],
       capsize=5, alpha=0.7, color='steelblue', edgecolor='black')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Test R² (Cross-Validation)', fontsize=12)
ax.set_title(f'{PLATFORM.title()} - Cross-Validation Performance', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{PLATFORM}_cv_performance.png', dpi=300, bbox_inches='tight')
print(f" Saved: {PLATFORM}_cv_performance.png")
plt.close()

# =============================================================================
# INTERPRETATION & RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 70)
print("INTERPRETATION & RECOMMENDATIONS")
print("=" * 70)

print(f"\n Dataset Characteristics:")
print(f"   Sample size: {len(df)}")
print(f"   Features: {len(feature_cols)}")
print(f"   Samples per feature: {len(df)/len(feature_cols):.1f}")

if len(df)/len(feature_cols) < 5:
    print(f"     WARNING: Very low samples-per-feature ratio!")
    print(f"   Recommendation: Use simpler models or feature selection")

print(f"\n Model Recommendations:")

# Check for overfitting
high_overfit = results_df[results_df['Overfitting Gap'] > 0.20]
if len(high_overfit) > 0:
    print(f"\n  Models with HIGH overfitting risk:")
    for _, row in high_overfit.iterrows():
        print(f"   • {row['Model']}: gap = {row['Overfitting Gap']:.4f}")
    print(f"   → Avoid these models in final dissertation")

# Recommend best models
good_models = results_df[
    (results_df['Test R² (mean)'] > 0.60) & 
    (results_df['Overfitting Gap'] < 0.15)
]

if len(good_models) > 0:
    print(f"\n Recommended models (good performance, low overfitting):")
    for _, row in good_models.head(3).iterrows():
        print(f"   • {row['Model']}")
        print(f"     Test R²: {row['Test R² (mean)']:.4f} (±{row['Test R² (std)']:.4f})")
        print(f"     Overfitting gap: {row['Overfitting Gap']:.4f}")

print(f"\n For Dissertation:")
print(f"   1. Report cross-validated Test R² (not train R²)")
print(f"   2. Include standard deviation (shows stability)")
print(f"   3. Discuss overfitting gap explicitly")
print(f"   4. Use simpler models for final results")
print(f"   5. Compare against baseline models")

print("\n" + "=" * 70)
print(" ROBUST VALIDATION COMPLETE!")
print("=" * 70)

print(f"\n Key Takeaways:")
print(f"   • Best CV Test R²: {best_model['Test R² (mean)']:.4f}")
print(f"   • This is the honest performance estimate")
print(f"   • Previous single-split results likely overestimated")
print(f"   • Use these CV results in dissertation")