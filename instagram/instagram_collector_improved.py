"""
Instagram Data Processor - For top_insta_influencers_data.csv
Handles special formatting: 3.3k, 475.8m, 29.0b, 1.39%
"""

import pandas as pd
import numpy as np
import re

print("=" * 70)
print("INSTAGRAM DATA PROCESSOR - Top 200 Influencers")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS TO PARSE SPECIAL FORMATS
# =============================================================================

def parse_number(value):
    """
    Convert strings like '3.3k', '475.8m', '29.0b' to numbers
    """
    if pd.isna(value) or value == '':
        return np.nan
    
    value_str = str(value).strip().lower()
    
    # Remove any commas
    value_str = value_str.replace(',', '')
    
    # Check for multipliers
    if 'b' in value_str:  # billions
        return float(value_str.replace('b', '')) * 1_000_000_000
    elif 'm' in value_str:  # millions
        return float(value_str.replace('m', '')) * 1_000_000
    elif 'k' in value_str:  # thousands
        return float(value_str.replace('k', '')) * 1_000
    else:
        try:
            return float(value_str)
        except:
            return np.nan

def parse_percentage(value):
    """
    Convert strings like '1.39%', 'NaN%' to decimal numbers
    """
    if pd.isna(value) or value == '' or value == 'NaN%':
        return np.nan
    
    value_str = str(value).strip().lower()
    
    # Remove % sign
    value_str = value_str.replace('%', '')
    
    try:
        return float(value_str) / 100  # Convert to decimal
    except:
        return np.nan

# =============================================================================
# LOAD DATA
# =============================================================================

CSV_FILE = 'top_insta_influencers_data.csv'

print(f"\n Loading {CSV_FILE}...")

try:
    # Read with encoding to handle special characters
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    print(f" Loaded successfully!")
except Exception as e:
    print(f" Error loading file: {e}")
    exit(1)

print(f"\n Dataset Info:")
print(f"   Total accounts: {len(df)}")
print(f"   Columns: {len(df.columns)}")

print(f"\n Columns:")
for col in df.columns:
    print(f"   • {col}")

print(f"\n First 3 rows (raw data):")
print(df.head(3))

# =============================================================================
# CONVERT SPECIAL FORMATS TO NUMBERS
# =============================================================================

print("\n" + "=" * 70)
print("CONVERTING FORMATS")
print("=" * 70)

# Convert posts (3.3k -> 3300)
print("\n Converting 'posts'...")
df['posts_numeric'] = df['posts'].apply(parse_number)
print(f"   Example: {df['posts'].iloc[0]} → {df['posts_numeric'].iloc[0]:,.0f}")

# Convert followers (475.8m -> 475800000)
print("\n Converting 'followers'...")
df['followers_numeric'] = df['followers'].apply(parse_number)
print(f"   Example: {df['followers'].iloc[0]} → {df['followers_numeric'].iloc[0]:,.0f}")

# Convert avg_likes (8.7m -> 8700000)
print("\n Converting 'avg_likes'...")
df['avg_likes_numeric'] = df['avg_likes'].apply(parse_number)
print(f"   Example: {df['avg_likes'].iloc[0]} → {df['avg_likes_numeric'].iloc[0]:,.0f}")

# Convert new_post_avg_like
print("\n Converting 'new_post_avg_like'...")
df['new_post_avg_like_numeric'] = df['new_post_avg_like'].apply(parse_number)
print(f"   Example: {df['new_post_avg_like'].iloc[0]} → {df['new_post_avg_like_numeric'].iloc[0]:,.0f}")

# Convert total_likes (29.0b -> 29000000000)
print("\n Converting 'total_likes'...")
df['total_likes_numeric'] = df['total_likes'].apply(parse_number)
print(f"   Example: {df['total_likes'].iloc[0]} → {df['total_likes_numeric'].iloc[0]:,.0f}")

# Convert engagement rate (1.39% -> 0.0139)
print("\n Converting '60_day_eng_rate'...")
df['engagement_rate'] = df['60_day_eng_rate'].apply(parse_percentage)
print(f"   Example: {df['60_day_eng_rate'].iloc[0]} → {df['engagement_rate'].iloc[0]:.4f}")

print("\n All conversions complete!")

# =============================================================================
# DATA QUALITY CHECK
# =============================================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\n Missing values in numeric columns:")
numeric_cols = ['followers_numeric', 'posts_numeric', 'avg_likes_numeric', 
                'engagement_rate', 'new_post_avg_like_numeric', 'total_likes_numeric']

for col in numeric_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"   {col}: {missing} missing ({missing/len(df)*100:.1f}%)")

# Fill missing engagement rates with 0
df['engagement_rate'] = df['engagement_rate'].fillna(0)

print("\n Filled missing engagement rates with 0")

# =============================================================================
# STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"\n Follower statistics:")
print(f"   Min: {df['followers_numeric'].min():,.0f} ({df['followers_numeric'].min()/1_000_000:.1f}M)")
print(f"   Max: {df['followers_numeric'].max():,.0f} ({df['followers_numeric'].max()/1_000_000:.1f}M)")
print(f"   Median: {df['followers_numeric'].median():,.0f} ({df['followers_numeric'].median()/1_000_000:.1f}M)")
print(f"   Mean: {df['followers_numeric'].mean():,.0f} ({df['followers_numeric'].mean()/1_000_000:.1f}M)")

print(f"\n Engagement rate statistics:")
print(f"   Min: {df['engagement_rate'].min():.4f} ({df['engagement_rate'].min()*100:.2f}%)")
print(f"   Max: {df['engagement_rate'].max():.4f} ({df['engagement_rate'].max()*100:.2f}%)")
print(f"   Median: {df['engagement_rate'].median():.4f} ({df['engagement_rate'].median()*100:.2f}%)")
print(f"   Mean: {df['engagement_rate'].mean():.4f} ({df['engagement_rate'].mean()*100:.2f}%)")

print(f"\n This is MEGA-INFLUENCER data!")
print(f"   Tier: Top-tier (32M-475M followers)")
print(f"   Perfect for comparing with YouTube mid-tier!")

# =============================================================================
# SELECT 35-40 ACCOUNTS
# =============================================================================

print("\n" + "=" * 70)
print("SELECTING 40 ACCOUNTS FOR ANALYSIS")
print("=" * 70)

# Strategy: Take top 40 by rank (most influential)
sample = df.head(40).copy()

print(f"\n Selected top 40 accounts by rank")
print(f"\nSelected accounts:")
print(f"   Rank 1: {sample.iloc[0]['channel_info']} ({sample.iloc[0]['followers_numeric']/1_000_000:.1f}M followers)")
print(f"   Rank 20: {sample.iloc[19]['channel_info']} ({sample.iloc[19]['followers_numeric']/1_000_000:.1f}M followers)")
print(f"   Rank 40: {sample.iloc[39]['channel_info']} ({sample.iloc[39]['followers_numeric']/1_000_000:.1f}M followers)")

# Create clean dataset with numeric values
final = pd.DataFrame({
    'username': sample['channel_info'],
    'rank': sample['rank'],
    'influence_score': sample['influence_score'],
    'posts': sample['posts_numeric'],
    'followers': sample['followers_numeric'],
    'avg_likes': sample['avg_likes_numeric'],
    'engagement_rate_60day': sample['engagement_rate'],
    'new_post_avg_like': sample['new_post_avg_like_numeric'],
    'total_likes': sample['total_likes_numeric'],
    'country': sample['country']
})

# =============================================================================
# SAVE RESULTS
# =============================================================================

final.to_csv('instagram_data_ready.csv', index=False)

print("\n" + "=" * 70)
print(" PROCESSING COMPLETE!")
print("=" * 70)

print(f"\n FINAL DATASET:")
print(f"   Accounts: {len(final)}")
print(f"   Columns: {len(final.columns)}")
print(f"   File: instagram_data_ready.csv")

print(f"\n SUMMARY STATISTICS:")
print(f"   Avg followers: {final['followers'].mean():,.0f} ({final['followers'].mean()/1_000_000:.1f}M)")
print(f"   Min followers: {final['followers'].min():,.0f} ({final['followers'].min()/1_000_000:.1f}M)")
print(f"   Max followers: {final['followers'].max():,.0f} ({final['followers'].max()/1_000_000:.1f}M)")
print(f"   Avg engagement rate: {final['engagement_rate_60day'].mean():.4f} ({final['engagement_rate_60day'].mean()*100:.2f}%)")
print(f"   Avg likes per post: {final['avg_likes'].mean():,.0f} ({final['avg_likes'].mean()/1_000_000:.2f}M)")

# Show tier distribution
print(f"\n FOLLOWER TIERS IN YOUR SAMPLE:")
bins = [0, 50_000_000, 100_000_000, 200_000_000, float('inf')]
labels = ['32M-50M', '50M-100M', '100M-200M', '200M+']
final['tier'] = pd.cut(final['followers'], bins=bins, labels=labels)
print(final['tier'].value_counts().sort_index())

print(f"\n FOR YOUR DISSERTATION:")
print("""
Multi-Tier Research Design:
  • YouTube: Mid-tier influencers (100K-1M followers, n=45)
  • Instagram: Mega influencers (50M-475M followers, n=40)
  
Research Value:
   Compares engagement effectiveness across scales
   Shows how metrics work at different tiers
   Reflects real brand partnership strategies
   More interesting than single-tier analysis!
  
Expected Finding:
  "Mid-tier influencers (YouTube) likely show higher engagement 
   rates than mega-influencers (Instagram) due to audience 
   intimacy effects, consistent with existing literature."
""")

print("\n" + "=" * 70)
print(" NEXT STEP")
print("=" * 70)

print("""
 You now have 40 Instagram mega-influencers ready!

NEXT: Calculate features

Run: python3 instagram_features_from_your_csv.py

This will create 12 features for ML models!
""")

print(" SUCCESS! Top 40 Instagram influencers processed and ready!")