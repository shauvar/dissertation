"""
Baseline Models
Simple heuristics to compare against ML models
Your ML models need to beat these baselines to be successful!
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=" * 70)
print("BASELINE MODELS - Simple Heuristics for Comparison")
print("=" * 70)

# Load data
df = pd.read_csv('youtube_features_final.csv')

print(f"\nLoaded {len(df)} channels")

# =============================================================================
# DEFINE TARGET VARIABLE
# =============================================================================

# For now, we'll use engagement_rate_mean as our target
# (You can change this later based on your research question)
TARGET = 'engagement_rate_mean'

print(f"\n Target Variable: {TARGET}")
print(f"   Mean: {df[TARGET].mean():.4f}")
print(f"   Std: {df[TARGET].std():.4f}")
print(f"   Min: {df[TARGET].min():.4f}")
print(f"   Max: {df[TARGET].max():.4f}")

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

# Split data (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"\n Data Split:")
print(f"   Training: {len(train_df)} channels")
print(f"   Testing: {len(test_df)} channels")

# =============================================================================
# BASELINE 1: MEAN PREDICTOR
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE 1: MEAN PREDICTOR")
print("=" * 70)
print("Strategy: Predict the mean engagement rate for everyone")

# Calculate mean from training data
mean_engagement = train_df[TARGET].mean()

# Make predictions
y_true_test = test_df[TARGET]
y_pred_mean = np.full(len(test_df), mean_engagement)

# Evaluate
rmse_mean = np.sqrt(mean_squared_error(y_true_test, y_pred_mean))
mae_mean = mean_absolute_error(y_true_test, y_pred_mean)
r2_mean = r2_score(y_true_test, y_pred_mean)

print(f"\n Performance:")
print(f"   R² Score: {r2_mean:.4f}")
print(f"   RMSE: {rmse_mean:.4f}")
print(f"   MAE: {mae_mean:.4f}")
print(f"   MAPE: {np.mean(np.abs((y_true_test - y_pred_mean) / y_true_test)) * 100:.2f}%")

# =============================================================================
# BASELINE 2: FOLLOWER COUNT HEURISTIC
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE 2: FOLLOWER COUNT HEURISTIC")
print("=" * 70)
print("Strategy: Bigger channels → Better engagement (common assumption)")

# Calculate correlation in training data
corr = train_df[['subscribers', TARGET]].corr().iloc[0, 1]
print(f"\nCorrelation between subscribers and {TARGET}: {corr:.4f}")

# Fit simple linear regression manually
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(
    train_df['subscribers'], 
    train_df[TARGET]
)

# Make predictions
y_pred_followers = slope * test_df['subscribers'] + intercept

# Evaluate
rmse_followers = np.sqrt(mean_squared_error(y_true_test, y_pred_followers))
mae_followers = mean_absolute_error(y_true_test, y_pred_followers)
r2_followers = r2_score(y_true_test, y_pred_followers)

print(f"\n Performance:")
print(f"   R² Score: {r2_followers:.4f}")
print(f"   RMSE: {rmse_followers:.4f}")
print(f"   MAE: {mae_followers:.4f}")
print(f"   MAPE: {np.mean(np.abs((y_true_test - y_pred_followers) / y_true_test)) * 100:.2f}%")

# =============================================================================
# BASELINE 3: VIEWS PER SUBSCRIBER HEURISTIC
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE 3: VIEWS PER SUBSCRIBER HEURISTIC")
print("=" * 70)
print("Strategy: High views/subscriber → Better engagement")

# Calculate correlation
corr = train_df[['views_per_subscriber', TARGET]].corr().iloc[0, 1]
print(f"\nCorrelation between views_per_subscriber and {TARGET}: {corr:.4f}")

# Fit simple linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    train_df['views_per_subscriber'], 
    train_df[TARGET]
)

# Make predictions
y_pred_views_per_sub = slope * test_df['views_per_subscriber'] + intercept

# Evaluate
rmse_vps = np.sqrt(mean_squared_error(y_true_test, y_pred_views_per_sub))
mae_vps = mean_absolute_error(y_true_test, y_pred_views_per_sub)
r2_vps = r2_score(y_true_test, y_pred_views_per_sub)

print(f"\n Performance:")
print(f"   R² Score: {r2_vps:.4f}")
print(f"   RMSE: {rmse_vps:.4f}")
print(f"   MAE: {mae_vps:.4f}")
print(f"   MAPE: {np.mean(np.abs((y_true_test - y_pred_views_per_sub) / y_true_test)) * 100:.2f}%")

# =============================================================================
# BASELINE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

baseline_results = pd.DataFrame({
    'Model': ['Mean Predictor', 'Follower Count', 'Views/Subscriber'],
    'R²': [r2_mean, r2_followers, r2_vps],
    'RMSE': [rmse_mean, rmse_followers, rmse_vps],
    'MAE': [mae_mean, mae_followers, mae_vps]
})

print("\n" + baseline_results.to_string(index=False))

# Find best baseline
best_baseline_idx = baseline_results['R²'].idxmax()
best_baseline = baseline_results.iloc[best_baseline_idx]

print(f"\n BEST BASELINE: {best_baseline['Model']}")
print(f"   R² = {best_baseline['R²']:.4f}")
print(f"   RMSE = {best_baseline['RMSE']:.4f}")

# Save results
baseline_results.to_csv('baseline_results.csv', index=False)
print(f"\n Saved: baseline_results.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================

# Plot baseline predictions vs actual
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Baseline 1: Mean
axes[0].scatter(y_true_test, y_pred_mean, alpha=0.6)
axes[0].plot([y_true_test.min(), y_true_test.max()], 
             [y_true_test.min(), y_true_test.max()], 
             'r--', lw=2)
axes[0].set_xlabel('Actual Engagement Rate')
axes[0].set_ylabel('Predicted Engagement Rate')
axes[0].set_title(f'Baseline 1: Mean (R²={r2_mean:.3f})')

# Baseline 2: Followers
axes[1].scatter(y_true_test, y_pred_followers, alpha=0.6, color='green')
axes[1].plot([y_true_test.min(), y_true_test.max()], 
             [y_true_test.min(), y_true_test.max()], 
             'r--', lw=2)
axes[1].set_xlabel('Actual Engagement Rate')
axes[1].set_ylabel('Predicted Engagement Rate')
axes[1].set_title(f'Baseline 2: Followers (R²={r2_followers:.3f})')

# Baseline 3: Views per Sub
axes[2].scatter(y_true_test, y_pred_views_per_sub, alpha=0.6, color='coral')
axes[2].plot([y_true_test.min(), y_true_test.max()], 
             [y_true_test.min(), y_true_test.max()], 
             'r--', lw=2)
axes[2].set_xlabel('Actual Engagement Rate')
axes[2].set_ylabel('Predicted Engagement Rate')
axes[2].set_title(f'Baseline 3: Views/Sub (R²={r2_vps:.3f})')

plt.tight_layout()
plt.savefig('baseline_predictions.png', dpi=300, bbox_inches='tight')
print(f" Saved: baseline_predictions.png")

# =============================================================================
# SUCCESS CRITERIA
# =============================================================================

print("\n" + "=" * 70)
print("SUCCESS CRITERIA FOR ML MODELS")
print("=" * 70)

target_r2 = best_baseline['R²'] * 1.3  # 30% improvement
target_rmse = best_baseline['RMSE'] * 0.85  # 15% reduction

print(f"\n Your ML models should achieve:")
print(f"   R² ≥ {target_r2:.4f} (30% improvement over best baseline)")
print(f"   RMSE ≤ {target_rmse:.4f} (15% reduction from best baseline)")

print(f"\n If your ML models can't beat these baselines,")
print(f"   then simple heuristics are more effective than complex ML!")

print("\n Next step: Run ml_models.py to train Random Forest, XGBoost, and LightGBM!")