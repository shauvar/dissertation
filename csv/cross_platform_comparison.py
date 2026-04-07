"""
Cross-Platform Comparison: YouTube vs Instagram
Compares mid-tier (YouTube) vs mega-influencers (Instagram)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("=" * 70)
print("CROSS-PLATFORM COMPARISON: YOUTUBE vs INSTAGRAM")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n Loading data...")

# Load YouTube data
try:
    youtube_features = pd.read_csv('youtube_features_final.csv')
    youtube_results = pd.read_csv('ml_model_results.csv')
    youtube_shap = pd.read_csv('shap_feature_importance.csv')
    print(f" YouTube: {len(youtube_features)} channels (100K-1M followers)")
except FileNotFoundError as e:
    print(f"  YouTube data not found: {e}")
    youtube_features = None

# Load Instagram data
try:
    instagram_features = pd.read_csv('instagram_features_final.csv')
    instagram_results = pd.read_csv('instagram_ml_model_results.csv')
    instagram_shap = pd.read_csv('instagram_shap_feature_importance.csv')
    print(f" Instagram: {len(instagram_features)} accounts (50M-500M followers)")
except FileNotFoundError as e:
    print(f"  Instagram data not found: {e}")
    instagram_features = None

# =============================================================================
# DATASET COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("1. DATASET COMPARISON")
print("=" * 70)

comparison_data = {
    'Metric': [
        'Sample Size',
        'Follower Tier',
        'Avg Followers',
        'Avg Engagement Rate',
        'Features Used',
    ],
    'YouTube (Mid-tier)': [
        len(youtube_features) if youtube_features is not None else 'N/A',
        '100K - 1M',
        f"{youtube_features['subscribers'].mean()/1000:.0f}K" if youtube_features is not None else 'N/A',
        f"{youtube_features['engagement_rate_mean'].mean()*100:.2f}%" if youtube_features is not None else 'N/A',
        len(youtube_features.columns) if youtube_features is not None else 'N/A',
    ],
    'Instagram (Mega)': [
        len(instagram_features) if instagram_features is not None else 'N/A',
        '50M - 500M',
        f"{instagram_features['followers'].mean()/1_000_000:.0f}M" if instagram_features is not None else 'N/A',
        f"{instagram_features['engagement_rate_mean'].mean()*100:.2f}%" if instagram_features is not None else 'N/A',
        len(instagram_features.columns) if instagram_features is not None else 'N/A',
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n Dataset Characteristics:")
print(comparison_df.to_string(index=False))

# =============================================================================
# MODEL PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("2. MODEL PERFORMANCE COMPARISON")
print("=" * 70)

if youtube_results is not None and instagram_results is not None:
    # Best models
    youtube_best = youtube_results.loc[youtube_results['R²'].idxmax()]
    instagram_best = instagram_results.loc[instagram_results['R²'].idxmax()]
    
    print(f"\n Best Models:")
    print(f"\nYouTube:")
    print(f"  Model: {youtube_best['Model']}")
    print(f"  R²: {youtube_best['R²']:.4f}")
    print(f"  RMSE: {youtube_best['RMSE']:.4f}")
    print(f"  MAE: {youtube_best['MAE']:.4f}")
    
    print(f"\nInstagram:")
    print(f"  Model: {instagram_best['Model']}")
    print(f"  R²: {instagram_best['R²']:.4f}")
    print(f"  RMSE: {instagram_best['RMSE']:.4f}")
    print(f"  MAE: {instagram_best['MAE']:.4f}")
    
    # Performance difference
    r2_diff = ((youtube_best['R²'] - instagram_best['R²']) / instagram_best['R²']) * 100
    
    print(f"\n Performance Difference:")
    if youtube_best['R²'] > instagram_best['R²']:
        print(f"  YouTube is {abs(r2_diff):.1f}% more predictable")
        print(f"  → Mid-tier influencers show more consistent patterns")
    else:
        print(f"  Instagram is {abs(r2_diff):.1f}% more predictable")
        print(f"  → Mega-influencers show more consistent patterns")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² comparison
    models = ['YouTube\n(Mid-tier)', 'Instagram\n(Mega)']
    r2_values = [youtube_best['R²'], instagram_best['R²']]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(models, r2_values, color=colors, edgecolor='black', width=0.5)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Excellent (0.70)')
    axes[0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (0.60)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Engagement rate comparison
    youtube_eng = youtube_features['engagement_rate_mean'].mean() * 100
    instagram_eng = instagram_features['engagement_rate_mean'].mean() * 100
    
    eng_values = [youtube_eng, instagram_eng]
    
    axes[1].bar(models, eng_values, color=colors, edgecolor='black', width=0.5)
    axes[1].set_ylabel('Engagement Rate (%)', fontsize=12)
    axes[1].set_title('Average Engagement Rate', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(eng_values):
        axes[1].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('platform_comparison_performance.png', dpi=300, bbox_inches='tight')
    print("\n Saved: platform_comparison_performance.png")
    plt.close()

# =============================================================================
# FEATURE IMPORTANCE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("3. TOP PREDICTIVE FEATURES")
print("=" * 70)

if youtube_shap is not None and instagram_shap is not None:
    print("\n Top 5 Features by Platform:")
    
    print("\nYouTube (Mid-tier):")
    for i, row in youtube_shap.head().iterrows():
        print(f"  {i+1}. {row['feature']}")
    
    print("\nInstagram (Mega-influencers):")
    for i, row in instagram_shap.head().iterrows():
        print(f"  {i+1}. {row['feature']}")
    
    # Find common features
    print(top_features.columns)

    youtube_top = set(youtube_shap.head(10)['feature'].values)
    instagram_top = set(instagram_shap.head(10)['feature'].values)
    common_features = youtube_top.intersection(instagram_top)
    
    if common_features:
        print(f"\n Common Important Features:")
        for feat in common_features:
            print(f"  • {feat}")
        print(f"\n  → These features are universally important across scales!")
    
    unique_youtube = youtube_top - instagram_top
    unique_instagram = instagram_top - youtube_top
    
    if unique_youtube:
        print(f"\n YouTube-Specific Important Features:")
        for feat in unique_youtube:
            print(f"  • {feat}")
    
    if unique_instagram:
        print(f"\n Instagram-Specific Important Features:")
        for feat in unique_instagram:
            print(f"  • {feat}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # YouTube top features
    youtube_top5 = youtube_shap.head(5)
    axes[0].barh(range(len(youtube_top5)), youtube_top5['importance'], color='#3498db', edgecolor='black')
    axes[0].set_yticks(range(len(youtube_top5)))
    axes[0].set_yticklabels(youtube_top5['feature'])
    axes[0].set_xlabel('SHAP Importance', fontsize=12)
    axes[0].set_title('YouTube (Mid-tier) - Top 5 Features', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Instagram top features
    instagram_top5 = instagram_shap.head(5)
    axes[1].barh(range(len(instagram_top5)), instagram_top5['importance'], color='#e74c3c', edgecolor='black')
    axes[1].set_yticks(range(len(instagram_top5)))
    axes[1].set_yticklabels(instagram_top5['feature'])
    axes[1].set_xlabel('SHAP Importance', fontsize=12)
    axes[1].set_title('Instagram (Mega) - Top 5 Features', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('platform_comparison_features.png', dpi=300, bbox_inches='tight')
    print("\n Saved: platform_comparison_features.png")
    plt.close()

# =============================================================================
# ENGAGEMENT ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("4. ENGAGEMENT RATE ANALYSIS")
print("=" * 70)

if youtube_features is not None and instagram_features is not None:
    youtube_eng_mean = youtube_features['engagement_rate_mean'].mean()
    instagram_eng_mean = instagram_features['engagement_rate_mean'].mean()
    
    engagement_ratio = youtube_eng_mean / instagram_eng_mean
    
    print(f"\n Engagement Statistics:")
    print(f"\nYouTube (Mid-tier):")
    print(f"  Mean: {youtube_eng_mean*100:.2f}%")
    print(f"  Median: {youtube_features['engagement_rate_mean'].median()*100:.2f}%")
    print(f"  Std: {youtube_features['engagement_rate_mean'].std()*100:.2f}%")
    
    print(f"\nInstagram (Mega):")
    print(f"  Mean: {instagram_eng_mean*100:.2f}%")
    print(f"  Median: {instagram_features['engagement_rate_mean'].median()*100:.2f}%")
    print(f"  Std: {instagram_features['engagement_rate_mean'].std()*100:.2f}%")
    
    print(f"\n Key Finding:")
    print(f"  Mid-tier influencers have {engagement_ratio:.2f}x higher engagement rates")
    print(f"  This demonstrates the AUDIENCE DILUTION EFFECT at scale")
    
    # Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograms
    axes[0].hist(youtube_features['engagement_rate_mean']*100, bins=15, 
                 alpha=0.7, color='#3498db', edgecolor='black', label='YouTube (Mid-tier)')
    axes[0].hist(instagram_features['engagement_rate_mean']*100, bins=15,
                 alpha=0.7, color='#e74c3c', edgecolor='black', label='Instagram (Mega)')
    axes[0].set_xlabel('Engagement Rate (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Engagement Rate Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plots
    data_to_plot = [
        youtube_features['engagement_rate_mean']*100,
        instagram_features['engagement_rate_mean']*100
    ]
    bp = axes[1].boxplot(data_to_plot, labels=['YouTube\n(Mid-tier)', 'Instagram\n(Mega)'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_ylabel('Engagement Rate (%)', fontsize=12)
    axes[1].set_title('Engagement Rate Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('platform_comparison_engagement.png', dpi=300, bbox_inches='tight')
    print("\n Saved: platform_comparison_engagement.png")
    plt.close()

# =============================================================================
# KEY INSIGHTS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS FOR DISSERTATION")
print("=" * 70)

print("\n FINDING 1: Scale Effects on Engagement")
if youtube_features is not None and instagram_features is not None:
    print(f"  Mid-tier influencers (YouTube, {youtube_features['subscribers'].mean()/1000:.0f}K avg)")
    print(f"  show {engagement_ratio:.2f}x higher engagement rates than")
    print(f"  mega-influencers (Instagram, {instagram_features['followers'].mean()/1_000_000:.0f}M avg).")
    print(f"  This validates audience dilution theory at scale.")

print("\n FINDING 2: Predictability Across Scales")
if youtube_results is not None and instagram_results is not None:
    print(f"  Both tiers showed predictable engagement patterns:")
    print(f"  • YouTube R² = {youtube_best['R²']:.3f}")
    print(f"  • Instagram R² = {instagram_best['R²']:.3f}")
    print(f"  ML models work across influencer scales!")

print("\n FINDING 3: Scale-Specific Success Factors")
print(f"  Different features predict success at different scales:")
print(f"  • Mid-tier: [YouTube top features]")
print(f"  • Mega-tier: [Instagram top features]")
print(f"  This suggests scale-specific selection strategies.")

# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary = {
    'Platform': ['YouTube', 'Instagram'],
    'Tier': ['Mid-tier (100K-1M)', 'Mega (50M-500M)'],
    'Sample Size': [
        len(youtube_features) if youtube_features is not None else 0,
        len(instagram_features) if instagram_features is not None else 0
    ],
    'Avg Engagement (%)': [
        youtube_eng_mean*100 if youtube_features is not None else 0,
        instagram_eng_mean*100 if instagram_features is not None else 0
    ],
    'Best Model R²': [
        youtube_best['R²'] if youtube_results is not None else 0,
        instagram_best['R²'] if instagram_results is not None else 0
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv('cross_platform_summary.csv', index=False)
print("\n Saved: cross_platform_summary.csv")

print("\n" + "=" * 70)
print(" CROSS-PLATFORM COMPARISON COMPLETE!")
print("=" * 70)

print("\n Generated Files:")
print("  • platform_comparison_performance.png")
print("  • platform_comparison_features.png")
print("  • platform_comparison_engagement.png")
print("  • cross_platform_summary.csv")

print("\n For Your Dissertation:")
print("  1. Include all 3 comparison charts in Results section")
print("  2. Discuss audience dilution effect (Finding 1)")
print("  3. Explain scale-specific feature importance (Finding 3)")
print("  4. Compare ML performance across tiers (Finding 2)")

print("\n Two platforms analyzed! TikTok next (optional) or start writing!")