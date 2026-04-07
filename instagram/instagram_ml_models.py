"""
Machine Learning Models - Instagram Mega Influencers
Random Forest, XGBoost, LightGBM with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_style("whitegrid")

print("=" * 70)
print("MACHINE LEARNING MODELS - Instagram Mega Influencers")
print("=" * 70)

# Load data
df = pd.read_csv('../csv/instagram_features_final.csv')
print(f"\n Loaded {len(df)} Instagram accounts with {len(df.columns)} columns")

# =============================================================================
# PREPARE DATA
# =============================================================================

print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# Target variable
target = 'engagement_rate_mean'

# Features to exclude
exclude_cols = [
    'username',  # identifier
    'rank',  # not a feature
    'country',  # categorical (could be encoded but skip for now)
    'engagement_rate_mean',  # target variable
    'engagement_rate_60day',  # same as target
]

# Select features
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

print(f"\n Features selected: {len(feature_cols)}")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i}. {feat}")

# Prepare X and y
X = df[feature_cols]
y = df[target]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n Data split:")
print(f"  Training set: {len(X_train)} accounts ({len(X_train)/len(df)*100:.0f}%)")
print(f"  Test set: {len(X_test)} accounts ({len(X_test)/len(df)*100:.0f}%)")

# =============================================================================
# MODEL 1: RANDOM FOREST
# =============================================================================

print("\n" + "=" * 70)
print("MODEL 1: RANDOM FOREST")
print("=" * 70)

print("\n Training Random Forest with GridSearchCV...")

# Define parameter grid
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(
    rf, 
    rf_param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

rf_grid.fit(X_train, y_train)

print(f" Best parameters: {rf_grid.best_params_}")

# Predictions
rf_pred = rf_grid.predict(X_test)

# Metrics
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100

print(f"\n Performance:")
print(f"  R²: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  MAE: {rf_mae:.4f}")
print(f"  MAPE: {rf_mape:.2f}%")

# Feature importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_grid.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n Top 5 Most Important Features:")
print(rf_importance.head().to_string(index=False))

# =============================================================================
# MODEL 2: XGBOOST
# =============================================================================

print("\n" + "=" * 70)
print("MODEL 2: XGBOOST")
print("=" * 70)

print("\n Training XGBoost with GridSearchCV...")

# Define parameter grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

# GridSearchCV
xgb = XGBRegressor(random_state=42, verbosity=0)
xgb_grid = GridSearchCV(
    xgb,
    xgb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

xgb_grid.fit(X_train, y_train)

print(f" Best parameters: {xgb_grid.best_params_}")

# Predictions
xgb_pred = xgb_grid.predict(X_test)

# Metrics
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100

print(f"\n Performance:")
print(f"  R²: {xgb_r2:.4f}")
print(f"  RMSE: {xgb_rmse:.4f}")
print(f"  MAE: {xgb_mae:.4f}")
print(f"  MAPE: {xgb_mape:.2f}%")

# Feature importance
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_grid.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n Top 5 Most Important Features:")
print(xgb_importance.head().to_string(index=False))

# =============================================================================
# MODEL 3: LIGHTGBM
# =============================================================================

print("\n" + "=" * 70)
print("MODEL 3: LIGHTGBM")
print("=" * 70)

print("\n Training LightGBM with GridSearchCV...")

# Define parameter grid
lgbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [15, 31, 63]
}

# GridSearchCV
lgbm = LGBMRegressor(random_state=42, verbosity=-1)
lgbm_grid = GridSearchCV(
    lgbm,
    lgbm_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

lgbm_grid.fit(X_train, y_train)

print(f" Best parameters: {lgbm_grid.best_params_}")

# Predictions
lgbm_pred = lgbm_grid.predict(X_test)

# Metrics
lgbm_r2 = r2_score(y_test, lgbm_pred)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
lgbm_mape = np.mean(np.abs((y_test - lgbm_pred) / y_test)) * 100

print(f"\n Performance:")
print(f"  R²: {lgbm_r2:.4f}")
print(f"  RMSE: {lgbm_rmse:.4f}")
print(f"  MAE: {lgbm_mae:.4f}")
print(f"  MAPE: {lgbm_mape:.2f}%")

# Feature importance
lgbm_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgbm_grid.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n Top 5 Most Important Features:")
print(lgbm_importance.head().to_string(index=False))

# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# Create comparison DataFrame
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'R²': [rf_r2, xgb_r2, lgbm_r2],
    'RMSE': [rf_rmse, xgb_rmse, lgbm_rmse],
    'MAE': [rf_mae, xgb_mae, lgbm_mae],
    'MAPE': [rf_mape, xgb_mape, lgbm_mape]
})

print("\n All Models:")
print(results.to_string(index=False))

# Save results
results.to_csv('instagram_ml_model_results.csv', index=False)
print("\n Saved: instagram_ml_model_results.csv")

# Find best model
best_model_idx = results['R²'].idxmax()
best_model_name = results.loc[best_model_idx, 'Model']
best_model_r2 = results.loc[best_model_idx, 'R²']

print(f"\n Best Model: {best_model_name}")
print(f"   R²: {best_model_r2:.4f}")

# Save best model
if best_model_name == 'Random Forest':
    best_model = rf_grid.best_estimator_
    best_pred = rf_pred
    best_importance = rf_importance
elif best_model_name == 'XGBoost':
    best_model = xgb_grid.best_estimator_
    best_pred = xgb_pred
    best_importance = xgb_importance
else:
    best_model = lgbm_grid.best_estimator_
    best_pred = lgbm_pred
    best_importance = lgbm_importance

# Save model and features
with open('instagram_best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('instagram_feature_list.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\n Saved best model: instagram_best_model.pkl")
print(f" Saved feature list: instagram_feature_list.pkl")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# 1. Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
results.plot(x='Model', y='R²', kind='bar', ax=axes[0], legend=False, 
             color=['skyblue', 'lightgreen', 'lightcoral'], edgecolor='black')
axes[0].set_title('Model Performance - R² Score', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_xlabel('Model')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# RMSE comparison
results.plot(x='Model', y='RMSE', kind='bar', ax=axes[1], legend=False,
             color=['skyblue', 'lightgreen', 'lightcoral'], edgecolor='black')
axes[1].set_title('Model Performance - RMSE', fontsize=14, fontweight='bold')
axes[1].set_ylabel('RMSE')
axes[1].set_xlabel('Model')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('instagram_model_comparison.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_model_comparison.png")
plt.close()

# 2. Feature importance
plt.figure(figsize=(10, 8))
top_n = min(10, len(best_importance))
top_features = best_importance.head(top_n)

plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title(f'Top {top_n} Most Important Features - {best_model_name}', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('instagram_feature_importance.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_feature_importance.png")
plt.close()

# 3. Predictions vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
axes[0].scatter(y_test * 100, best_pred * 100, alpha=0.6, s=100, color='steelblue')
axes[0].plot([y_test.min()*100, y_test.max()*100],
             [y_test.min()*100, y_test.max()*100],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Engagement Rate (%)', fontsize=12)
axes[0].set_ylabel('Predicted Engagement Rate (%)', fontsize=12)
axes[0].set_title(f'{best_model_name} Predictions vs Actual', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test - best_pred
axes[1].scatter(best_pred * 100, residuals * 100, alpha=0.6, s=100, color='coral')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Engagement Rate (%)', fontsize=12)
axes[1].set_ylabel('Residuals (%)', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('instagram_model_predictions.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_model_predictions.png")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print(" ML TRAINING COMPLETE!")
print("=" * 70)

print(f"\n Best Model: {best_model_name}")
print(f"  • R²: {best_model_r2:.4f} ({best_model_r2*100:.1f}% variance explained)")
print(f"  • Features used: {len(feature_cols)}")

print(f"\n Top 5 Predictive Features:")
for i, row in best_importance.head().iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

# Compare with baseline
try:
    baseline_results = pd.read_csv('instagram_baseline_results.csv')
    best_baseline_r2 = baseline_results['R²'].max()
    improvement = ((best_model_r2 - best_baseline_r2) / best_baseline_r2) * 100
    
    print(f"\n Improvement over baseline:")
    print(f"  Best baseline R²: {best_baseline_r2:.4f}")
    print(f"  ML model R²: {best_model_r2:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    if improvement > 30:
        print(f"   EXCELLENT! Significantly better than baselines!")
    elif improvement > 15:
        print(f"   GOOD! Notable improvement over baselines")
    else:
        print(f"    Modest improvement - consider feature engineering")
except:
    print("\n Run instagram_baseline_models.py to compare with baselines")

print(f"\n Generated Files:")
print(f"  • instagram_ml_model_results.csv")
print(f"  • instagram_best_model.pkl")
print(f"  • instagram_feature_list.pkl")
print(f"  • instagram_model_comparison.png")
print(f"  • instagram_feature_importance.png")
print(f"  • instagram_model_predictions.png")

print(f"\n Next step: Run instagram_shap_analysis.py for interpretability!")