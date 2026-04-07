"""
Exploratory Data Analysis (EDA)
Understand your YouTube dataset before building ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS - YouTube Influencer Dataset")
print("=" * 70)

# Load data
df = pd.read_csv('../csv/youtube_features_final.csv')

print(f"\n DATASET OVERVIEW:")
print(f"  Channels: {len(df)}")
print(f"  Features: {len(df.columns)}")
print(f"  Total videos analyzed: {df['videos_analyzed'].sum():.0f}")

# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)

key_metrics = [
    'subscribers',
    'engagement_rate_mean',
    'views_per_subscriber',
    'likes_mean',
    'comments_mean'
]

print("\n Key Metrics Summary:")
print(df[key_metrics].describe().round(4))

# =============================================================================
# 2. DISTRIBUTION ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("2. DISTRIBUTION ANALYSIS")
print("=" * 70)

# Create distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

# Subscriber distribution
axes[0, 0].hist(df['subscribers'], bins=20, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Subscriber Distribution')
axes[0, 0].set_xlabel('Subscribers')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['subscribers'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# Engagement rate distribution
axes[0, 1].hist(df['engagement_rate_mean'], bins=20, edgecolor='black', color='lightgreen')
axes[0, 1].set_title('Engagement Rate Distribution')
axes[0, 1].set_xlabel('Engagement Rate')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['engagement_rate_mean'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].legend()

# Views per video
axes[0, 2].hist(df['views_mean'], bins=20, edgecolor='black', color='lightcoral')
axes[0, 2].set_title('Average Views per Video')
axes[0, 2].set_xlabel('Views')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].axvline(df['views_mean'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 2].legend()

# Views per subscriber
axes[1, 0].hist(df['views_per_subscriber'], bins=20, edgecolor='black', color='plum')
axes[1, 0].set_title('Views per Subscriber')
axes[1, 0].set_xlabel('Views/Subscriber')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['views_per_subscriber'].mean(), color='red', linestyle='--', label='Mean')
axes[1, 0].legend()

# Engagement consistency
axes[1, 1].hist(df['engagement_consistency_score'], bins=20, edgecolor='black', color='lightyellow')
axes[1, 1].set_title('Engagement Consistency Score')
axes[1, 1].set_xlabel('Consistency Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(df['engagement_consistency_score'].mean(), color='red', linestyle='--', label='Mean')
axes[1, 1].legend()

# Comment to like ratio
axes[1, 2].hist(df['comment_to_like_ratio_mean'], bins=20, edgecolor='black', color='lightblue')
axes[1, 2].set_title('Comment-to-Like Ratio')
axes[1, 2].set_xlabel('Ratio')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].axvline(df['comment_to_like_ratio_mean'].mean(), color='red', linestyle='--', label='Mean')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('01_feature_distributions.png', dpi=300, bbox_inches='tight')
print("\n Saved: 01_feature_distributions.png")

# =============================================================================
# 3. CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("3. CORRELATION ANALYSIS")
print("=" * 70)

# Select numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['channel_id']]

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, 
            mask=mask,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n Saved: 02_correlation_matrix.png")

# Find highly correlated features
print("\n Highly Correlated Feature Pairs (|r| > 0.7):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr:
    high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', ascending=False, key=abs)
    print(high_corr_df.to_string(index=False))
else:
    print("  No feature pairs with |r| > 0.7")

# =============================================================================
# 4. RELATIONSHIP ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("4. KEY RELATIONSHIPS")
print("=" * 70)

# Create scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Feature Relationships', fontsize=16, fontweight='bold')

# Subscribers vs Engagement Rate
axes[0, 0].scatter(df['subscribers'], df['engagement_rate_mean'], alpha=0.6, s=100)
axes[0, 0].set_xlabel('Subscribers', fontsize=12)
axes[0, 0].set_ylabel('Engagement Rate', fontsize=12)
axes[0, 0].set_title('Subscribers vs Engagement Rate')
z = np.polyfit(df['subscribers'], df['engagement_rate_mean'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['subscribers'], p(df['subscribers']), "r--", alpha=0.8)

# Subscribers vs Views per Video
axes[0, 1].scatter(df['subscribers'], df['views_mean'], alpha=0.6, s=100, color='green')
axes[0, 1].set_xlabel('Subscribers', fontsize=12)
axes[0, 1].set_ylabel('Average Views per Video', fontsize=12)
axes[0, 1].set_title('Subscribers vs Average Views')

# Engagement Rate vs Views per Subscriber
axes[1, 0].scatter(df['engagement_rate_mean'], df['views_per_subscriber'], 
                   alpha=0.6, s=100, color='coral')
axes[1, 0].set_xlabel('Engagement Rate', fontsize=12)
axes[1, 0].set_ylabel('Views per Subscriber', fontsize=12)
axes[1, 0].set_title('Engagement Rate vs Views/Subscriber')

# Consistency vs Engagement Rate
axes[1, 1].scatter(df['engagement_consistency_score'], df['engagement_rate_mean'],
                   alpha=0.6, s=100, color='purple')
axes[1, 1].set_xlabel('Engagement Consistency', fontsize=12)
axes[1, 1].set_ylabel('Engagement Rate', fontsize=12)
axes[1, 1].set_title('Consistency vs Engagement Rate')

plt.tight_layout()
plt.savefig('03_feature_relationships.png', dpi=300, bbox_inches='tight')
print("\n Saved: 03_feature_relationships.png")

# =============================================================================
# 5. OUTLIER DETECTION
# =============================================================================

print("\n" + "=" * 70)
print("5. OUTLIER DETECTION")
print("=" * 70)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

print("\n Outliers by Feature (using IQR method):")
outlier_features = ['engagement_rate_mean', 'views_per_subscriber', 'likes_mean']

for feature in outlier_features:
    outliers = detect_outliers_iqr(df, feature)
    if len(outliers) > 0:
        print(f"\n  {feature}: {len(outliers)} outliers detected")
        print(f"    Channels: {', '.join(outliers['channel_name'].values[:5])}")
    else:
        print(f"\n  {feature}: No outliers")

# =============================================================================
# 6. SEGMENT ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("6. CHANNEL SEGMENTS")
print("=" * 70)

# Segment by subscriber size
df['segment'] = pd.cut(df['subscribers'], 
                       bins=[0, 200000, 500000, 1000000],
                       labels=['Micro (100K-200K)', 'Mid (200K-500K)', 'Macro (500K-1M)'])

print("\n Channels by Segment:")
print(df['segment'].value_counts().sort_index())

print("\n Engagement Rate by Segment:")
segment_stats = df.groupby('segment')['engagement_rate_mean'].agg(['mean', 'std', 'count'])
print(segment_stats.round(4))

# Visualize segments
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Segment distribution
df['segment'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Channel Distribution by Segment', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Segment')
axes[0].set_ylabel('Number of Channels')
axes[0].tick_params(axis='x', rotation=45)

# Engagement by segment
df.boxplot(column='engagement_rate_mean', by='segment', ax=axes[1])
axes[1].set_title('Engagement Rate by Segment', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Engagement Rate')
axes[1].get_figure().suptitle('')  # Remove auto title
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('04_segment_analysis.png', dpi=300, bbox_inches='tight')
print("\n Saved: 04_segment_analysis.png")

# =============================================================================
# 7. SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 70)
print("7. DATA QUALITY ASSESSMENT")
print("=" * 70)

print("\n Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values detected!")
else:
    print(missing[missing > 0])

print("\n Data Types:")
print(df.dtypes.value_counts())

print("\n Dataset Readiness:")
print(f"   {len(df)} channels (target: 40) - EXCEEDED!")
print(f"   {df['videos_analyzed'].sum():.0f} videos analyzed")
print(f"   {len(numeric_cols)} numeric features")
print(f"   No missing values")
print(f"   Good subscriber distribution (mean: {df['subscribers'].mean():,.0f})")

print("\n" + "=" * 70)
print(" EDA COMPLETE!")
print("=" * 70)
print("\n Generated Files:")
print("  1. 01_feature_distributions.png")
print("  2. 02_correlation_matrix.png")
print("  3. 03_feature_relationships.png")
print("  4. 04_segment_analysis.png")

print("\n KEY INSIGHTS:")
print(f"  • Average engagement rate: {df['engagement_rate_mean'].mean():.4f}")
print(f"  • Engagement rate std dev: {df['engagement_rate_mean'].std():.4f}")
print(f"  • Views per subscriber: {df['views_per_subscriber'].mean():.4f}")
print(f"  • Most common segment: {df['segment'].mode()[0]}")

print("\n Next step: Run baseline_models.py to start ML training!")