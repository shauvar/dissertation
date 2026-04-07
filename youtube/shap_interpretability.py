"""
SHAP Interpretability Analysis
Explain model predictions using SHAP values
Critical for your research - shows WHICH features matter most!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SHAP INTERPRETABILITY ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. LOAD MODEL AND DATA
# =============================================================================

print("\n Loading model and data...")

# Load best model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature list
with open('feature_list.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Load data
df = pd.read_csv('youtube_features_final.csv')
X = df[feature_cols]

print(f" Model loaded: {type(model).__name__}")
print(f" Features: {len(feature_cols)}")
print(f" Samples: {len(X)}")

# =============================================================================
# 2. COMPUTE SHAP VALUES
# =============================================================================

print("\n" + "=" * 70)
print("2. COMPUTING SHAP VALUES")
print("=" * 70)
print("\nThis may take 1-2 minutes...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

print(" SHAP values computed!")

# =============================================================================
# 3. GLOBAL FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 70)
print("3. GLOBAL FEATURE IMPORTANCE")
print("=" * 70)

# Calculate mean absolute SHAP values
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP Importance': shap_importance
}).sort_values('SHAP Importance', ascending=False)

print("\n Top 10 Most Important Features (by SHAP):")
print(feature_importance_df.head(10).to_string(index=False))

# Save feature importance
feature_importance_df.to_csv('shap_feature_importance.csv', index=False)
print("\n Saved: shap_feature_importance.csv")

# =============================================================================
# 4. SHAP SUMMARY PLOT
# =============================================================================

print("\n" + "=" * 70)
print("4. SHAP SUMMARY PLOTS")
print("=" * 70)

# Summary plot (bee swarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
print(" Saved: shap_summary_beeswarm.png")
plt.close()

# Summary plot (bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
print(" Saved: shap_summary_bar.png")
plt.close()

# =============================================================================
# 5. TOP 5 FEATURES ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("5. TOP 5 FEATURES DETAILED ANALYSIS")
print("=" * 70)

top_5_features = feature_importance_df.head(5)['Feature'].values

print("\n Analyzing top 5 predictive features:")
for i, feature in enumerate(top_5_features, 1):
    print(f"  {i}. {feature}")

# Dependence plots for top 5 features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_5_features):
    feature_idx = feature_cols.index(feature)
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        ax=axes[idx],
        show=False
    )
    axes[idx].set_title(f'{feature}')

# Hide last subplot if not used
if len(top_5_features) < 6:
    axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
print("\n Saved: shap_dependence_plots.png")
plt.close()

# =============================================================================
# 6. INDIVIDUAL PREDICTIONS EXPLANATION
# =============================================================================

print("\n" + "=" * 70)
print("6. INDIVIDUAL PREDICTION EXPLANATIONS")
print("=" * 70)

# Select 3 example channels (low, medium, high engagement)
y_actual = df['engagement_rate_mean']
low_idx = y_actual.idxmin()
mid_idx = (y_actual - y_actual.median()).abs().idxmin()
high_idx = y_actual.idxmax()

examples = {
    'Low Engagement': low_idx,
    'Medium Engagement': mid_idx,
    'High Engagement': high_idx
}

print("\n Explaining predictions for 3 example channels:")

for label, idx in examples.items():
    channel_name = df.loc[idx, 'channel_name']
    actual_engagement = y_actual.iloc[idx]
    predicted_engagement = model.predict(X.iloc[[idx]])[0]
    
    print(f"\n{label}:")
    print(f"  Channel: {channel_name}")
    print(f"  Actual: {actual_engagement:.4f}")
    print(f"  Predicted: {predicted_engagement:.4f}")
    print(f"  Error: {abs(actual_engagement - predicted_engagement):.4f}")

# Force plots for individual predictions
for label, idx in examples.items():
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    filename = f"shap_force_plot_{label.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}")
    plt.close()

# =============================================================================
# 7. INSIGHTS REPORT
# =============================================================================

print("\n" + "=" * 70)
print("7. KEY INSIGHTS FOR DISSERTATION")
print("=" * 70)

# Get top 3-5 features as specified in research plan
top_features = feature_importance_df.head(5)

print("\n TOP 5 PREDICTIVE FEATURES:")
for i, row in top_features.iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['SHAP Importance']:.4f}")

print("\n INTERPRETATION GUIDELINES:")
print("  • Higher SHAP value → Feature increases engagement prediction")
print("  • Lower SHAP value → Feature decreases engagement prediction")
print("  • SHAP values show CONTRIBUTION, not just correlation")

print("\n FOR YOUR DISSERTATION:")
print("  Write: 'SHAP analysis identified the top 5 predictive metrics:")
for i, row in top_features.iterrows():
    print(f"    {i+1}. {row['Feature']}")
print("  These features collectively explain X% of model predictions.'")

# Calculate cumulative importance
top_features['Cumulative %'] = (
    top_features['SHAP Importance'].cumsum() / 
    top_features['SHAP Importance'].sum() * 100
)

print(f"\n Top 5 features explain {top_features['Cumulative %'].iloc[-1]:.1f}% of predictions")

# =============================================================================
# 8. CROSS-PLATFORM COMPARISON (FOR FUTURE)
# =============================================================================

print("\n" + "=" * 70)
print("8. PLATFORM-SPECIFIC INSIGHTS")
print("=" * 70)

print("\n Current Analysis: YouTube")
print("  Top predictor: " + top_features.iloc[0]['Feature'])
print("  This suggests brands should prioritize this metric on YouTube")

print("\n FUTURE WORK:")
print("  • Repeat this analysis for Instagram and TikTok")
print("  • Compare top features across platforms")
print("  • Identify platform-specific vs universal predictors")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print(" SHAP ANALYSIS COMPLETE!")
print("=" * 70)

print("\n Generated Files:")
print("  1. shap_feature_importance.csv")
print("  2. shap_summary_beeswarm.png")
print("  3. shap_summary_bar.png")
print("  4. shap_dependence_plots.png")
print("  5. shap_force_plot_low_engagement.png")
print("  6. shap_force_plot_medium_engagement.png")
print("  7. shap_force_plot_high_engagement.png")

print("\n FOR YOUR DISSERTATION:")
print("\n  METHODOLOGY Section:")
print("    'We employed SHAP (SHapley Additive exPlanations) to interpret")
print("     model predictions and identify the most influential features.'")
print("\n  RESULTS Section:")
print("    'SHAP analysis revealed that [TOP FEATURE] was the strongest")
print("     predictor of engagement-based effectiveness on YouTube.'")
print("    Include: shap_summary_beeswarm.png and shap_dependence_plots.png")
print("\n  DISCUSSION Section:")
print("    'Unlike black-box models, our approach provides actionable insights:")
print("     brands can prioritize [TOP 3 FEATURES] when selecting influencers.'")

print("\n Next steps:")
print("  1. Add these visualizations to your dissertation")
print("  2. Write interpretation of top 5 features")
print("  3. Consider creating dashboard prototype (optional)")
print("  4. Start Instagram/TikTok data collection")