"""
ML Models - Random Forest, XGBoost, LightGBM
Train interpretable tree-based models for engagement prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ML MODELS - Engagement Prediction")
print("=" * 70)

# Load data
df = pd.read_csv('youtube_features_final.csv')
print(f"\n Loaded {len(df)} channels with {len(df.columns)} features")

# =============================================================================
# 1. FEATURE SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("1. FEATURE SELECTION")
print("=" * 70)

# Define target
TARGET = 'engagement_rate_mean'

# Features to exclude (identifiers and target)
exclude_features = [
    'channel_id', 'channel_name', 'channel_url', 
    'created_at', 'country', 'description',
    TARGET,  # Don't use target as feature!
    'engagement_rate_std',  # Too closely related to target
]

# Select features
feature_cols = [col for col in df.columns if col not in exclude_features]
print(f"\n Using {len(feature_cols)} features:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i}. {feat}")

# Prepare X and y
X = df[feature_cols]
y = df[TARGET]

print(f"\n Target Variable: {TARGET}")
print(f"   Mean: {y.mean():.4f}")
print(f"   Range: [{y.min():.4f}, {y.max():.4f}]")

# =============================================================================
# 2. TRAIN-TEST SPLIT
# =============================================================================

print("\n" + "=" * 70)
print("2. TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("3. MODEL TRAINING")
print("=" * 70)

models = {}
results = []

# -----------------------------------------------------------------------------
# MODEL 1: RANDOM FOREST
# -----------------------------------------------------------------------------

print("\n Training Random Forest...")

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(
    rf, rf_params, cv=5, scoring='r2', 
    n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)

# Best model
rf_best = rf_grid.best_estimator_
models['Random Forest'] = rf_best

# Predictions
y_pred_rf = rf_best.predict(X_test)

# Metrics
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mape = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

results.append({
    'Model': 'Random Forest',
    'R²': rf_r2,
    'RMSE': rf_rmse,
    'MAE': rf_mae,
    'MAPE': rf_mape,
    'Best Params': rf_grid.best_params_
})

print(f"   Best params: {rf_grid.best_params_}")
print(f"   R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}")

# -----------------------------------------------------------------------------
# MODEL 2: XGBOOST
# -----------------------------------------------------------------------------

print("\n Training XGBoost...")

xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42, verbosity=0)
xgb_grid = GridSearchCV(
    xgb, xgb_params, cv=5, scoring='r2',
    n_jobs=-1, verbose=0
)
xgb_grid.fit(X_train, y_train)

# Best model
xgb_best = xgb_grid.best_estimator_
models['XGBoost'] = xgb_best

# Predictions
y_pred_xgb = xgb_best.predict(X_test)

# Metrics
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

results.append({
    'Model': 'XGBoost',
    'R²': xgb_r2,
    'RMSE': xgb_rmse,
    'MAE': xgb_mae,
    'MAPE': xgb_mape,
    'Best Params': xgb_grid.best_params_
})

print(f"   Best params: {xgb_grid.best_params_}")
print(f"   R² = {xgb_r2:.4f}, RMSE = {xgb_rmse:.4f}")

# -----------------------------------------------------------------------------
# MODEL 3: LIGHTGBM
# -----------------------------------------------------------------------------

print("\n Training LightGBM...")

lgbm_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [15, 31, 63]
}

lgbm = LGBMRegressor(random_state=42, verbosity=-1)
lgbm_grid = GridSearchCV(
    lgbm, lgbm_params, cv=5, scoring='r2',
    n_jobs=-1, verbose=0
)
lgbm_grid.fit(X_train, y_train)

# Best model
lgbm_best = lgbm_grid.best_estimator_
models['LightGBM'] = lgbm_best

# Predictions
y_pred_lgbm = lgbm_best.predict(X_test)

# Metrics
lgbm_r2 = r2_score(y_test, y_pred_lgbm)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
lgbm_mae = mean_absolute_error(y_test, y_pred_lgbm)
lgbm_mape = np.mean(np.abs((y_test - y_pred_lgbm) / y_test)) * 100

results.append({
    'Model': 'LightGBM',
    'R²': lgbm_r2,
    'RMSE': lgbm_rmse,
    'MAE': lgbm_mae,
    'MAPE': lgbm_mape,
    'Best Params': lgbm_grid.best_params_
})

print(f"   Best params: {lgbm_grid.best_params_}")
print(f"   R² = {lgbm_r2:.4f}, RMSE = {lgbm_rmse:.4f}")

# =============================================================================
# 4. MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("4. MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)

print("\n" + results_df[['Model', 'R²', 'RMSE', 'MAE', 'MAPE']].to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_r2 = results_df.iloc[0]['R²']

print(f"\n BEST MODEL: {best_model_name}")
print(f"   R² = {best_r2:.4f}")

# Save results
results_df.to_csv('ml_model_results.csv', index=False)
print(f"\n Saved: ml_model_results.csv")

# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 70)
print("5. FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance from best model
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n Top 10 Most Important Features ({best_model_name}):")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved: feature_importance.png")

# =============================================================================
# 6. PREDICTIONS VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

model_names = ['Random Forest', 'XGBoost', 'LightGBM']
predictions = [y_pred_rf, y_pred_xgb, y_pred_lgbm]
r2_scores = [rf_r2, xgb_r2, lgbm_r2]

for idx, (name, pred, r2) in enumerate(zip(model_names, predictions, r2_scores)):
    axes[idx].scatter(y_test, pred, alpha=0.6)
    axes[idx].plot([y_test.min(), y_test.max()], 
                   [y_test.min(), y_test.max()], 
                   'r--', lw=2)
    axes[idx].set_xlabel('Actual Engagement Rate')
    axes[idx].set_ylabel('Predicted Engagement Rate')
    axes[idx].set_title(f'{name} (R²={r2:.3f})')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
print(f" Saved: model_predictions.png")

# =============================================================================
# 7. SAVE BEST MODEL
# =============================================================================

print("\n" + "=" * 70)
print("7. SAVE BEST MODEL")
print("=" * 70)

# Save model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save feature list
with open('feature_list.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\n Saved:")
print(f"  - best_model.pkl ({best_model_name})")
print(f"  - feature_list.pkl")

# =============================================================================
# 8. BASELINE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("8. COMPARISON WITH BASELINES")
print("=" * 70)

# Load baseline results
try:
    baseline_df = pd.read_csv('baseline_results.csv')
    best_baseline_r2 = baseline_df['R²'].max()
    
    print(f"\n Best Baseline R²: {best_baseline_r2:.4f}")
    print(f" Best ML Model R²: {best_r2:.4f}")
    
    improvement = ((best_r2 - best_baseline_r2) / best_baseline_r2) * 100
    print(f"\n Improvement: {improvement:.1f}%")
    
    if best_r2 >= 0.75:
        print(f"\n SUCCESS! R² ≥ 0.75 target achieved!")
    else:
        print(f"\n  R² = {best_r2:.4f} (target: ≥ 0.75)")
        print(f"   Consider: more data, feature engineering, or different target variable")
        
except FileNotFoundError:
    print("\n  Run baseline_models.py first for comparison")

print("\n" + "=" * 70)
print(" ML MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nNext step: Run shap_interpretability.py for model explanations!")