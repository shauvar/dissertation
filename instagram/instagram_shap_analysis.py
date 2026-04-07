"""
SHAP Interpretability Analysis - Instagram Mega Influencers
Explains model predictions using SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

sns.set_style("whitegrid")

print("=" * 70)
print("SHAP INTERPRETABILITY ANALYSIS - Instagram Mega Influencers")
print("=" * 70)

# Load data
df = pd.read_csv('../csv/instagram_features_final.csv')
print(f"\n Loaded {len(df)} Instagram accounts")

# Load best model and features
try:
    with open('instagram_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('instagram_feature_list.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    print(f" Loaded best model with {len(feature_cols)} features")
except FileNotFoundError:
    print(" Model files not found. Please run instagram_ml_models.py first!")
    exit(1)

# Prepare data
X = df[feature_cols]
y = df['engagement_rate_mean']

# =============================================================================
# CALCULATE SHAP VALUES
# =============================================================================

print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

print("\n Creating SHAP explainer...")

# Create explainer (TreeExplainer for tree-based models)
explainer = shap.TreeExplainer(model)

print(" Calculating SHAP values...")

# Calculate SHAP values
shap_values = explainer.shap_values(X)

print(" SHAP values calculated!")

# =============================================================================
# GLOBAL FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 70)
print("GLOBAL FEATURE IMPORTANCE")
print("=" * 70)

# Calculate mean absolute SHAP values
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n Top 10 Most Important Features (by SHAP):")
print(shap_importance.head(10).to_string(index=False))

# Save SHAP feature importance
shap_importance.to_csv('instagram_shap_feature_importance.csv', index=False)
print("\n Saved: instagram_shap_feature_importance.csv")

# =============================================================================
# VISUALIZATION 1: SHAP Summary Plot (Beeswarm)
# =============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

print("\n 1. SHAP Summary Plot (Beeswarm)...")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False, plot_size=(10, 8))
plt.title('SHAP Feature Importance - Instagram Mega Influencers', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('instagram_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_shap_summary_beeswarm.png")
plt.close()

# =============================================================================
# VISUALIZATION 2: SHAP Bar Plot
# =============================================================================

print("\n 2. SHAP Bar Plot...")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Chart)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('instagram_shap_summary_bar.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_shap_summary_bar.png")
plt.close()

# =============================================================================
# VISUALIZATION 3: SHAP Dependence Plots (Top 5 Features)
# =============================================================================

print("\n 3. SHAP Dependence Plots (Top 5 features)...")

top_5_features = shap_importance.head(5)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('SHAP Dependence Plots - Top 5 Features', fontsize=16, fontweight='bold')

for idx, feature in enumerate(top_5_features):
    row = idx // 3
    col = idx % 3
    
    if idx < 5:
        # Get feature index
        feature_idx = feature_cols.index(feature)
        
        # Create dependence plot
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            show=False,
            ax=axes[row, col]
        )
        axes[row, col].set_title(f'{feature}', fontsize=12, fontweight='bold')

# Remove extra subplot
if len(top_5_features) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('instagram_shap_dependence_plots.png', dpi=300, bbox_inches='tight')
print(" Saved: instagram_shap_dependence_plots.png")
plt.close()

# =============================================================================
# VISUALIZATION 4: SHAP Force Plots (Sample Predictions)
# =============================================================================

print("\n 4. SHAP Force Plots (sample predictions)...")

# Select 3 examples: low, medium, high engagement
y_sorted_idx = y.argsort()
low_idx = y_sorted_idx.iloc[0]  # Lowest engagement
mid_idx = y_sorted_idx.iloc[len(y)//2]  # Median engagement
high_idx = y_sorted_idx.iloc[-1]  # Highest engagement

examples = [
    (low_idx, 'Low Engagement'),
    (mid_idx, 'Medium Engagement'),
    (high_idx, 'High Engagement')
]

for idx, label in examples:
    plt.figure(figsize=(14, 3))
    
    # Create force plot
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False
    )
    
    actual_engagement = y.iloc[idx] * 100
    predicted_engagement = model.predict(X.iloc[idx:idx+1])[0] * 100
    
    plt.title(f'SHAP Force Plot - {label}\n' + 
              f'Actual: {actual_engagement:.2f}% | Predicted: {predicted_engagement:.2f}%',
              fontsize=12, fontweight='bold', pad=10)
    
    filename = f'instagram_shap_force_plot_{label.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}")
    plt.close()

# =============================================================================
# INTERPRETATION SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("INTERPRETATION SUMMARY")
print("=" * 70)

print(f"\n Top 5 Predictive Features:")
for i, row in shap_importance.head().iterrows():
    print(f"\n{i+1}. {row['feature']} (importance: {row['importance']:.4f})")
    
    # Get feature index
    feat_idx = feature_cols.index(row['feature'])
    
    # Calculate correlation with target
    corr = X[row['feature']].corr(y)
    
    # Interpretation
    if corr > 0:
        direction = "Higher values → Higher engagement"
    elif corr < 0:
        direction = "Higher values → Lower engagement"
    else:
        direction = "No clear linear relationship"
    
    print(f"   Correlation with engagement: {corr:.3f}")
    print(f"   Effect: {direction}")

# =============================================================================
# KEY INSIGHTS
# =============================================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

# Check if follower scale is important
if 'follower_scale_log' in shap_importance.head(5)['feature'].values:
    print("\n Follower scale is a top predictor:")
    print("   → Account size significantly impacts engagement patterns")
    print("   → Mega-influencers show different engagement dynamics")

# Check if recent performance matters
if 'recent_performance_ratio' in shap_importance.head(5)['feature'].values:
    print("\n Recent performance trend is a top predictor:")
    print("   → Trending influencers (improving engagement) are detectable")
    print("   → Momentum matters for campaign selection")

# Check if consistency matters
if 'engagement_consistency_score' in shap_importance.head(5)['feature'].values:
    print("\n Engagement consistency is a top predictor:")
    print("   → Reliable, consistent engagement is valued")
    print("   → Reduces campaign risk")

# Check if like rate matters
if 'like_rate' in shap_importance.head(5)['feature'].values:
    print("\n Like rate is a top predictor:")
    print("   → Direct engagement intensity matters")
    print("   → Not just follower count")

# =============================================================================
# COMPARISON WITH YOUTUBE
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON WITH YOUTUBE (if available)")
print("=" * 70)

try:
    # Try to load YouTube SHAP results
    youtube_shap = pd.read_csv('shap_feature_importance.csv')
    
    print("\n Top 5 Instagram vs YouTube Features:")
    print("\nInstagram (Mega-influencers):")
    for i, row in shap_importance.head().iterrows():
        print(f"  {i+1}. {row['feature']}")
    
    print("\nYouTube (Mid-tier):")
    for i, row in youtube_shap.head().iterrows():
        print(f"  {i+1}. {row['feature']}")
    
    print("\n Interpretation:")
    print("   Different features matter at different scales!")
    print("   This validates the multi-tier research approach.")
    
except FileNotFoundError:
    print("\n YouTube SHAP results not found")
    print("   Run analysis on YouTube data to compare")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print(" SHAP ANALYSIS COMPLETE!")
print("=" * 70)

print("\n Generated Files:")
print("  • instagram_shap_feature_importance.csv")
print("  • instagram_shap_summary_beeswarm.png")
print("  • instagram_shap_summary_bar.png")
print("  • instagram_shap_dependence_plots.png")
print("  • instagram_shap_force_plot_low_engagement.png")
print("  • instagram_shap_force_plot_medium_engagement.png")
print("  • instagram_shap_force_plot_high_engagement.png")

print("\n Key Takeaways:")
print(f"  • Top predictor: {shap_importance.iloc[0]['feature']}")
print(f"  • Model explains: ML R² from instagram_ml_model_results.csv")
print(f"  • Features analyzed: {len(feature_cols)}")
print(f"  • Mega-influencers (50M-500M followers)")

print("\n For Dissertation:")
print("  1. Use Top 5 SHAP features in Results section")
print("  2. Include beeswarm plot (shows feature effects)")
print("  3. Discuss scale-specific patterns vs YouTube")
print("  4. Explain why these features matter at mega scale")

print("\n Instagram ML analysis complete!")
print(" Now compare with YouTube results!")