"""
Calculate Features Script
Computes engagement metrics and aggregates to channel level
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("FEATURE CALCULATION")
print("=" * 60)

# Load data
try:
    videos_df = pd.read_csv('youtube_videos_raw.csv')
    channels_df = pd.read_csv('discovered_channels_100k_1m.csv')
    print(f"\n Loaded {len(videos_df)} videos from {len(channels_df)} channels")
except FileNotFoundError as e:
    print(f"\n Error: {e}")
    print("Please run collect_starter_channels_fixed.py and collect_video_data_fixed.py first.")
    exit(1)

print("\nCalculating features...")

# ============================================================
# VIDEO-LEVEL FEATURES
# ============================================================

# 1. Engagement rate (basic)
videos_df['engagement_rate'] = (videos_df['likes'] + videos_df['comments']) / videos_df['views'].replace(0, 1)

# 2. Like-to-view ratio
videos_df['like_to_view_ratio'] = videos_df['likes'] / videos_df['views'].replace(0, 1)

# 3. Comment-to-view ratio
videos_df['comment_to_view_ratio'] = videos_df['comments'] / videos_df['views'].replace(0, 1)

# 4. Comment-to-like ratio (deeper engagement indicator)
videos_df['comment_to_like_ratio'] = videos_df['comments'] / videos_df['likes'].replace(0, 1)

print(" Calculated video-level features")

# ============================================================
# CHANNEL-LEVEL AGGREGATIONS
# ============================================================

print(" Aggregating to channel level...")

channel_features = videos_df.groupby('channel_id').agg({
    # View metrics
    'views': ['mean', 'median', 'std', 'min', 'max'],
    
    # Engagement metrics
    'likes': ['mean', 'median', 'std'],
    'comments': ['mean', 'median', 'std'],
    
    # Engagement rates
    'engagement_rate': ['mean', 'std'],
    'like_to_view_ratio': ['mean', 'std'],
    'comment_to_view_ratio': ['mean', 'std'],
    'comment_to_like_ratio': ['mean'],
    
    # Content metrics
    'video_id': 'count',  # Number of videos analyzed
    'tags_count': 'mean'  # Average tags per video
}).reset_index()

# Flatten column names
channel_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                            for col in channel_features.columns.values]

# Rename for clarity
channel_features.rename(columns={
    'video_id_count': 'videos_analyzed',
    'tags_count_mean': 'avg_tags_per_video'
}, inplace=True)

print(f" Created {len(channel_features.columns)} aggregated features")

# ============================================================
# MERGE WITH CHANNEL DATA
# ============================================================

final_df = channels_df.merge(channel_features, on='channel_id', how='left')

# ============================================================
# DERIVED FEATURES
# ============================================================

print(" Computing derived features...")

# 1. Views per subscriber (relative reach)
final_df['views_per_subscriber'] = final_df['views_mean'] / final_df['subscribers']

# 2. Engagement consistency (lower std = more consistent)
final_df['engagement_consistency_score'] = 1 / (1 + final_df['engagement_rate_std'])

# 3. Subscriber-to-video ratio (how many videos per sub)
final_df['subs_per_video'] = final_df['subscribers'] / final_df['video_count']

# 4. Average performance per video
final_df['avg_views_per_video'] = final_df['total_views'] / final_df['video_count']

# 5. Recent activity level (videos analyzed / total videos)
final_df['recent_activity_rate'] = final_df['videos_analyzed'] / final_df['video_count']

print(f" Added 5 derived features")

# ============================================================
# CLEAN UP AND ORGANIZE
# ============================================================

# Reorder columns for better readability
column_order = [
    # Identifiers
    'channel_id',
    'channel_name',
    'channel_url',
    
    # Basic stats
    'subscribers',
    'video_count',
    'total_views',
    'videos_analyzed',
    
    # Engagement Quality (most important for your research!)
    'engagement_rate_mean',
    'engagement_rate_std',
    'engagement_consistency_score',
    'like_to_view_ratio_mean',
    'comment_to_view_ratio_mean',
    'comment_to_like_ratio_mean',
    
    # View metrics
    'views_mean',
    'views_median',
    'views_std',
    'views_per_subscriber',
    'avg_views_per_video',
    
    # Likes & Comments
    'likes_mean',
    'likes_median',
    'comments_mean',
    'comments_median',
    
    # Other
    'avg_tags_per_video',
    'subs_per_video',
    'recent_activity_rate',
    'created_at',
    'country'
]

# Keep only columns that exist
column_order = [col for col in column_order if col in final_df.columns]
final_df = final_df[column_order]

# ============================================================
# SAVE RESULTS
# ============================================================

final_df.to_csv('youtube_features_final.csv', index=False)

print("\n" + "=" * 60)
print("FEATURE CALCULATION COMPLETE!")
print("=" * 60)
print(f"\n DATASET SUMMARY:")
print(f"  Channels: {len(final_df)}")
print(f"  Features per channel: {len(final_df.columns)}")
print(f"  Total videos analyzed: {final_df['videos_analyzed'].sum():.0f}")

print(f"\n FILE SAVED:")
print(f"  - youtube_features_final.csv")

print(f"\n KEY METRICS (AVERAGE):")
print(f"  Subscribers: {final_df['subscribers'].mean():,.0f}")
print(f"  Videos: {final_df['video_count'].mean():.0f}")
print(f"  Engagement rate: {final_df['engagement_rate_mean'].mean():.4f}")
print(f"  Views per video: {final_df['views_mean'].mean():,.0f}")
print(f"  Views per subscriber: {final_df['views_per_subscriber'].mean():.4f}")

print(f"\n FEATURES AVAILABLE FOR ML:")
feature_list = [col for col in final_df.columns 
                if col not in ['channel_id', 'channel_name', 'channel_url', 'created_at', 'country']]
for i, feature in enumerate(feature_list, 1):
    print(f"  {i}. {feature}")

print(f"\n You now have {len(feature_list)} features ready for machine learning!")
print(f" Next step: Start building your ML models! ")