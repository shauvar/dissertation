"""
Baseline Models - Instagram Mega Influencers
Simple heuristics to compare against ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("=" * 70)
print("BASELINE MODELS - Instagram Mega Influencers")
print("=" * 70)

# Load data
df = pd.read_csv('../csv/instagram_features_final.csv')
print(f"\n Loaded {len(df)} Instagram accounts")

# Target variable
target = 'engagement_rate_mean'

# Prepare data
y = df[target]

# Train-test split (80/20)
X = df[['followers']]  # Simple feature for baseline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n Dataset split:")
print(f"  Training set: {len(X_train)} accounts")
print(f"  Test set: {len(X_test)} accounts")

# =============================================================================
# BASELINE 1: Mean Predictor
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE 1: MEAN PREDICTOR")
print("=" * 70)

# Predict mean engagement rate for all
mean_pred = np.full(len(y_test), y_train.mean())

r2_mean = r2_score(y_test, mean_pred)
rmse_mean = np.sqrt(mean_squared_error(y_test, mean_pred))
mae_mean = mean_absolute_error(y_test, mean_pred)
mape_mean = np.mean(np.abs((y_test - mean_pred) / y_test)) * 100

print(f"\n Performance:")
print(f"  R²: {r2_mean:.4f}")
print(f"  RMSE: {rmse_mean:.4f}")
print(f"  MAE: {mae_mean:.4f}")
print(f"  MAPE: {mape_mean:.2f}%")

# =============================================================================
# BASELINE 2: Follower-based Linear Model
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE 2: FOLLOWER-BASED LINEAR MODEL")
print("=" * 70)

from sklearn.linear_model import LinearRegression

# Simple linear regression: engagement ~ followers
lr_followers = LinearRegression()
lr_followers.fit(X_train, y_train)
y_pred_followers = lr_followers.predict(X_test)

r2_followers = r2_score(y_test, y_pred_followers)
rmse_followers = np.sqrt(mean_squared_error(y_test, y_pred_followers))
mae_followers = mean_absolute_error(y_test, y_pred_followers)
mape_followers = np.mean(np.abs((y_test - y_pred_followers) / y_test)) * 100

print(f"\n Performance:")
print(f"  R²: {r2_followers:.4f}")
print(f"  RMSE: {rmse_followers:.4f}")
print(f"  MAE: {mae_followers:.4f}")
print(f"  MAPE: {mape_followers:.2f}%")

# =============================================================================
# BASELINE 3: Like Rate Predictor
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE 3: LIKE RATE-BASED LINEAR MODEL")
print("=" * 70)

# Use like_rate as predictor
X_like = df[['like_rate']]
X_train_like, X_test_like, y_train_like, y_test_like = train_test_split(
    X_like, y, test_size=0.2, random_state=42
)

lr_like = LinearRegression()
lr_like.fit(X_train_like, y_train_like)
y_pred_like = lr_like.predict(X_test_like)

r2_like = r2_score(y_test_like, y_pred_like)
rmse_like = np.sqrt(mean_squared_error(y_test_like, y_pred_like))
mae_like = mean_absolute_error(y_test_like, y_pred_like)
mape_like = np.mean(np.abs((y_test_like - y_pred_like) / y_test_like)) * 100

print(f"\n Performance:")
print(f"  R²: {r2_like:.4f}")
print(f"  RMSE: {rmse_like:.4f}")
print(f"  MAE: {mae_like:.4f}")
print(f"  MAPE: {mape_like:.2f}%")

# =============================================================================
# COMPARISON & VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

# Create comparison DataFrame
baselines = pd.DataFrame({
    'Model': ['Mean Predictor', 'Follower-based', 'Like Rate-based'],
    'R²': [r2_mean, r2_followers, r2_like],
    'RMSE': [rmse_mean, rmse_followers, rmse_like],
    'MAE': [mae_mean, mae_followers, mae_like],
    'MAPE': [mape_mean, mape_followers, mape_like]
})

print("\n All Baselines:")
print(baselines.to_string(index=False))

# Save results
baselines.to_csv('instagram_baseline_results.csv', index=False)
print("\n Saved: instagram_baseline_results.csv")

# Find best baseline
best_baseline = baselines.loc[baselines['R²'].idxmax()]
print(f"\n Best Baseline: {best_baseline['Model']}")
print(f"   R²: {best_baseline['R²']:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
baselines.plot(x='Model', y='R²', kind='bar', ax=axes[0], legend=False, color='skyblue', edgecolor='black')
axes[0].set_title('Baseline Models - R² Score', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_xlabel('Model')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].tick_params(axis='x', rotation=45)

# Predictions vs Actual (best baseline)
if best_baseline['Model'] == 'Like Rate-based':
    best_pred = y_pred_like
    best_test = y_test_like
elif best_baseline['Model'] == 'Follower-based':
    best_pred = y_pred_followers
    best_test = y_test
else:
    best_pred = mean_pred
    best_test = y_test

axes[1].scatter(best_test * 100, best_pred * 100, alpha=0.6, s=100)
axes[1].plot([best_test.min()*100, best_test.max()*100], 
             [best_test.min()*100, best_test.max()*100], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Engagement Rate (%)', fontsize=12)
axes[1].set_ylabel('Predicted Engagement Rate (%)', fontsize=12)
axes[1].set_title(f'Best Baseline: {best_baseline["Model"]}', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('instagram_baseline_predictions.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_baseline_predictions.png")
plt.close()

# =============================================================================
# SUCCESS CRITERIA
# =============================================================================

print("\n" + "=" * 70)
print("SUCCESS CRITERIA FOR ML MODELS")
print("=" * 70)

best_r2 = baselines['R²'].max()
target_improvement = 0.30  # 30% improvement

ml_target_r2 = best_r2 * (1 + target_improvement)

print(f"\n ML Model Targets:")
print(f"  Best baseline R²: {best_r2:.4f}")
print(f"  Target improvement: {target_improvement*100:.0f}%")
print(f"  Minimum ML R²: {ml_target_r2:.4f}")

if ml_target_r2 > 0.75:
    print(f"\n  → ML models should achieve R² > 0.75 for excellent performance")
elif ml_target_r2 > 0.65:
    print(f"\n  → ML models should achieve R² > 0.65 for good performance")
else:
    print(f"\n  → ML models should achieve R² > {ml_target_r2:.2f} to beat baselines")

print("\n" + "=" * 70)
print(" BASELINE ANALYSIS COMPLETE!")
print("=" * 70)

print("\n Summary:")
print(f"  • Best baseline: {best_baseline['Model']} (R² = {best_r2:.4f})")
print(f"  • ML models need R² > {ml_target_r2:.4f} to show improvement")
print(f"  • Expected ML performance: R² = 0.60-0.75 for mega-influencers")

print("\n Next step: Run instagram_ml_models.py")