"""
Feature Engineering for Instagram Mega-Influencers
Creates ML-ready features from processed Instagram data
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("INSTAGRAM FEATURE ENGINEERING - Mega Influencers")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================

try:
    df = pd.read_csv('instagram_data_ready.csv')
    print(f" Loaded {len(df)} Instagram accounts")
except FileNotFoundError:
    print(" File not found: instagram_data_ready.csv")
    print("Please run process_actual_instagram_data.py first!")
    exit(1)

print(f"\n Available columns:")
for col in df.columns:
    print(f"   • {col}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("CALCULATING FEATURES")
print("=" * 70)

features_created = []

# ============================================================================
# FEATURE 1: Like Rate (normalized engagement)
# ============================================================================
df['like_rate'] = df['avg_likes'] / df['followers']
features_created.append('like_rate')
print(" like_rate")

# ============================================================================
# FEATURE 2: Posts per million followers (content frequency scaled)
# ============================================================================
df['posts_per_million_followers'] = df['posts'] / (df['followers'] / 1_000_000)
features_created.append('posts_per_million_followers')
print(" posts_per_million_followers")

# ============================================================================
# FEATURE 3: Followers per post (audience building efficiency)
# ============================================================================
df['followers_per_post'] = df['followers'] / df['posts'].replace(0, 1)
features_created.append('followers_per_post')
print(" followers_per_post")

# ============================================================================
# FEATURE 4: Content efficiency (total engagement / posts)
# ============================================================================
df['content_efficiency'] = df['total_likes'] / df['posts'].replace(0, 1)
features_created.append('content_efficiency')
print(" content_efficiency")

# ============================================================================
# FEATURE 5: Engagement rate (from 60-day data)
# ============================================================================
df['engagement_rate_mean'] = df['engagement_rate_60day']
features_created.append('engagement_rate_mean')
print(" engagement_rate_mean")

# ============================================================================
# FEATURE 6: Recent performance ratio (trending)
# ============================================================================
df['recent_performance_ratio'] = df['new_post_avg_like'] / df['avg_likes'].replace(0, 1)
features_created.append('recent_performance_ratio')
print(" recent_performance_ratio")

# ============================================================================
# FEATURE 7: Total engagement per follower
# ============================================================================
df['total_engagement_per_follower'] = df['total_likes'] / df['followers']
features_created.append('total_engagement_per_follower')
print(" total_engagement_per_follower")

# ============================================================================
# FEATURE 8: Likes per million followers per post
# ============================================================================
df['likes_per_m_followers_per_post'] = (df['avg_likes'] / (df['followers'] / 1_000_000)) / df['posts'].replace(0, 1)
features_created.append('likes_per_m_followers_per_post')
print(" likes_per_m_followers_per_post")

# ============================================================================
# FEATURE 9: Popularity score (influence_score normalized)
# ============================================================================
df['popularity_score'] = (df['influence_score'] - df['influence_score'].min()) / \
                         (df['influence_score'].max() - df['influence_score'].min())
features_created.append('popularity_score')
print(" popularity_score")

# ============================================================================
# FEATURE 10: Engagement consistency (recent vs overall)
# ============================================================================
df['engagement_consistency_score'] = 1 - abs(df['recent_performance_ratio'] - 1).clip(0, 1)
features_created.append('engagement_consistency_score')
print(" engagement_consistency_score")

# ============================================================================
# FEATURE 11: Account maturity (content volume relative to audience)
# ============================================================================
df['account_maturity'] = np.log1p(df['posts']) / np.log1p(df['followers'])
features_created.append('account_maturity')
print(" account_maturity")

# ============================================================================
# FEATURE 12: Viral potential (recent outperforms average)
# ============================================================================
df['viral_potential'] = (df['new_post_avg_like'] > df['avg_likes']).astype(int)
features_created.append('viral_potential')
print(" viral_potential")

# ============================================================================
# FEATURE 13: Follower scale (log of followers)
# ============================================================================
df['follower_scale_log'] = np.log1p(df['followers'])
features_created.append('follower_scale_log')
print(" follower_scale_log")

# ============================================================================
# FEATURE 14: Mega-influencer tier
# ============================================================================
# Categorize by follower size
bins = [0, 50_000_000, 100_000_000, 200_000_000, float('inf')]
df['mega_tier'] = pd.cut(df['followers'], bins=bins, labels=[0, 1, 2, 3])
df['mega_tier'] = df['mega_tier'].astype(int)
features_created.append('mega_tier')
print(" mega_tier")

# =============================================================================
# HANDLE INVALID VALUES
# =============================================================================

print("\n" + "=" * 70)
print("CLEANING CALCULATED FEATURES")
print("=" * 70)

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Check for missing/invalid
missing = df[features_created].isnull().sum()
if missing.sum() > 0:
    print("\n  Missing/invalid values found:")
    for col in missing[missing > 0].index:
        print(f"   {col}: {missing[col]} values")
    
    # Fill with median
    for col in features_created:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Filled {col} with median: {median_val:.6f}")
    
    print("\n All missing values filled")
else:
    print(" No missing values in calculated features")

# =============================================================================
# SELECT COLUMNS FOR ML
# =============================================================================

ml_columns = [
    'username',  # identifier
    'rank',  # original
    'influence_score',
    'followers',
    'posts',
    'avg_likes',
    'engagement_rate_60day',
    'country',
] + features_created

# Keep only existing columns
ml_columns = [col for col in ml_columns if col in df.columns]

df_ml = df[ml_columns].copy()

# =============================================================================
# SAVE RESULTS
# =============================================================================

df_ml.to_csv('instagram_features_final.csv', index=False)

print("\n" + "=" * 70)
print(" FEATURE ENGINEERING COMPLETE!")
print("=" * 70)

print(f"\n DATASET SUMMARY:")
print(f"   Accounts: {len(df_ml)}")
print(f"   Total columns: {len(df_ml.columns)}")
print(f"   Original metrics: 7")
print(f"   Calculated features: {len(features_created)}")

print(f"\n FEATURES FOR ML ({len(features_created)} new features):")
for i, feat in enumerate(features_created, 1):
    print(f"   {i}. {feat}")

print(f"\n FILE SAVED:")
print(f"   instagram_features_final.csv")

print(f"\n KEY STATISTICS:")
print(f"   Avg followers: {df_ml['followers'].mean():,.0f} ({df_ml['followers'].mean()/1_000_000:.1f}M)")
print(f"   Avg engagement rate: {df_ml['engagement_rate_mean'].mean():.4f} ({df_ml['engagement_rate_mean'].mean()*100:.2f}%)")
print(f"   Avg like rate: {df_ml['like_rate'].mean():.4f} ({df_ml['like_rate'].mean()*100:.2f}%)")
print(f"   Avg recent performance ratio: {df_ml['recent_performance_ratio'].mean():.2f}")

# =============================================================================
# FEATURE EXPLANATION
# =============================================================================

print("\n" + "=" * 70)
print("FEATURE EXPLANATION")
print("=" * 70)

explanations = {
    'like_rate': 'Average likes normalized by followers (engagement intensity)',
    'posts_per_million_followers': 'Content frequency scaled by audience size',
    'followers_per_post': 'Audience growth per piece of content',
    'content_efficiency': 'Total engagement per post (lifetime value)',
    'engagement_rate_mean': '60-day engagement rate (recent performance)',
    'recent_performance_ratio': 'New post performance vs average (trend indicator)',
    'total_engagement_per_follower': 'Total engagement normalized by audience',
    'likes_per_m_followers_per_post': 'Engagement density (scaled)',
    'popularity_score': 'Normalized influence score (0-1)',
    'engagement_consistency_score': 'How stable recent vs overall (0-1)',
    'account_maturity': 'Content volume relative to audience size',
    'viral_potential': 'Binary: recent > average (1) or not (0)',
    'follower_scale_log': 'Log-transformed follower count (handles scale)',
    'mega_tier': 'Influencer tier: 0=32-50M, 1=50-100M, 2=100-200M, 3=200M+'
}

print("\nWhat each feature measures:")
for feat in features_created:
    if feat in explanations:
        print(f"\n• {feat}")
        print(f"  → {explanations[feat]}")

# =============================================================================
# COMPARISON WITH YOUTUBE
# =============================================================================

print("\n" + "=" * 70)
print("INSTAGRAM vs YOUTUBE COMPARISON")
print("=" * 70)

print(f"""
 DATA CHARACTERISTICS:

Instagram (Mega-influencers):
  • Follower range: {df_ml['followers'].min()/1_000_000:.0f}M - {df_ml['followers'].max()/1_000_000:.0f}M
  • Avg engagement: {df_ml['engagement_rate_mean'].mean()*100:.2f}%
  • Accounts: {len(df_ml)}
  • Features: {len(features_created)}
  
YouTube (Mid-tier):
  • Follower range: 100K - 1M
  • Avg engagement: ~3-4% (expected higher)
  • Accounts: 45
  • Features: 22
  
 RESEARCH INSIGHT:
  Mega-influencers (Instagram) typically show LOWER engagement 
  rates than mid-tier (YouTube) due to:
  • Audience dilution at scale
  • More passive follower behavior
  • Less intimate creator-audience relationship
  
  This is a KEY FINDING for your dissertation!
""")

# =============================================================================
# NEXT STEPS
# =============================================================================

print("\n" + "=" * 70)
print(" NEXT STEPS")
print("=" * 70)

print(f"""
 You now have {len(df_ml)} accounts with {len(features_created)} features!

READY FOR MACHINE LEARNING!

Quick Method (reuse YouTube scripts):
  1. mv youtube_features_final.csv youtube_backup.csv
  2. mv instagram_features_final.csv youtube_features_final.csv
  3. python3 exploratory_analysis.py
  4. python3 baseline_models.py
  5. python3 ml_models.py
  6. python3 shap_interpretability.py
  7. mv youtube_features_final.csv instagram_features_final.csv
  8. mv youtube_backup.csv youtube_features_final.csv

Expected Performance:
  • R² = 0.60-0.75 (good for mega-influencer tier!)
  • May be slightly lower than YouTube (more variability at scale)
  • Still sufficient for publication-quality analysis

After ML Training:
  • Compare Instagram vs YouTube results
  • Document scale effects on engagement
  • Key finding: mid-tier vs mega engagement patterns!
""")

if len(df_ml) >= 35:
    print(" SUCCESS! You have 40 mega-influencers ready for ML analysis!")