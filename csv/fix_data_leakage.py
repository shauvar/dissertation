"""
Identify and Fix Data Leakage Issues
Checks for features that perfectly predict the target
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("=" * 70)
print("DATA LEAKAGE DETECTION & FIX")
print("=" * 70)

# Load data
PLATFORM = 'instagram'  # Change to 'instagram' as needed

if PLATFORM == 'youtube':
    df = pd.read_csv('youtube_features_final.csv')
else:
    df = pd.read_csv('instagram_features_final.csv')

print(f"\n Loaded {PLATFORM} data: {len(df)} samples")

# Target variable
target = 'engagement_rate_mean'

# Check all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

print(f"\n All numeric columns: {len(numeric_cols)}")

# =============================================================================
# IDENTIFY LEAKAGE
# =============================================================================

print("\n" + "=" * 70)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 70)

print("\n Correlation with target (engagement_rate_mean):")

correlations = []
for col in numeric_cols:
    if col != target and df[col].notna().sum() > 0:
        try:
            corr, pval = pearsonr(df[col].fillna(0), df[target])
            correlations.append({
                'Feature': col,
                'Correlation': abs(corr),
                'Sign': 'positive' if corr > 0 else 'negative'
            })
        except:
            pass

corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)

print("\n HIGH CORRELATION FEATURES (potential leakage):")
high_corr = corr_df[corr_df['Correlation'] > 0.95]
if len(high_corr) > 0:
    print(high_corr.to_string(index=False))
    print("\n  These features likely cause perfect prediction!")
else:
    print("  No obvious leakage detected")

print("\n  MODERATE CORRELATION FEATURES (check these):")
mod_corr = corr_df[(corr_df['Correlation'] > 0.80) & (corr_df['Correlation'] <= 0.95)]
if len(mod_corr) > 0:
    print(mod_corr.to_string(index=False))

print("\n SAFE FEATURES (correlation < 0.80):")
safe_features = corr_df[corr_df['Correlation'] <= 0.80]
print(f"  Found {len(safe_features)} safe features")
print(safe_features.head(15).to_string(index=False))

# =============================================================================
# IDENTIFY MULTICOLLINEARITY
# =============================================================================

print("\n" + "=" * 70)
print("CHECKING MULTICOLLINEARITY")
print("=" * 70)

# Calculate correlation matrix
exclude_cols = ['channel_name', 'channel_id', 'username', 'rank', 'country', target]
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

if len(feature_cols) > 0:
    feature_corr = df[feature_cols].corr()
    
    # Find highly correlated pairs
    high_pairs = []
    for i in range(len(feature_corr.columns)):
        for j in range(i+1, len(feature_corr.columns)):
            if abs(feature_corr.iloc[i, j]) > 0.90:
                high_pairs.append({
                    'Feature 1': feature_corr.columns[i],
                    'Feature 2': feature_corr.columns[j],
                    'Correlation': feature_corr.iloc[i, j]
                })
    
    if len(high_pairs) > 0:
        print(f"\n HIGHLY CORRELATED FEATURE PAIRS (>0.90):")
        pairs_df = pd.DataFrame(high_pairs).sort_values('Correlation', ascending=False, key=abs)
        print(pairs_df.to_string(index=False))
        print("\n  Keep only ONE feature from each pair!")
    else:
        print("\n No highly correlated feature pairs found")

# =============================================================================
# RECOMMENDED FEATURE SET
# =============================================================================

print("\n" + "=" * 70)
print("RECOMMENDED FEATURE SET")
print("=" * 70)

# Remove obvious leakage features
remove_features = []

# Check for features with "engagement" in name
for col in feature_cols:
    if 'engagement' in col.lower() and col != target:
        remove_features.append(col)
        print(f"  Removing '{col}' - contains 'engagement'")

# Remove features with near-perfect correlation
for _, row in high_corr.iterrows():
    if row['Feature'] not in remove_features:
        remove_features.append(row['Feature'])
        print(f"  Removing '{row['Feature']}' - correlation {row['Correlation']:.3f}")

# Create clean feature set
clean_features = [f for f in feature_cols if f not in remove_features]

# Limit to reasonable number (top 10 by importance or correlation)
if len(clean_features) > 10:
    # Keep top 10 moderately correlated features
    top_features = safe_features.head(10)['Feature'].tolist()
    clean_features = [f for f in clean_features if f in top_features]
    print(f"\n Limiting to top 10 features (samples-per-feature ratio)")

print(f"\n RECOMMENDED FEATURE SET ({len(clean_features)} features):")
for i, feat in enumerate(clean_features, 1):
    corr_value = corr_df[corr_df['Feature'] == feat]['Correlation'].values[0]
    print(f"  {i}. {feat} (r={corr_value:.3f})")

print(f"\n New samples-per-feature ratio: {len(df)/len(clean_features):.1f}")
if len(df)/len(clean_features) >= 4:
    print(f"    Good ratio (>4)")
elif len(df)/len(clean_features) >= 3:
    print(f"     Acceptable ratio (>3)")
else:
    print(f"     Still low - consider fewer features")

# =============================================================================
# CREATE CLEAN DATASET
# =============================================================================

print("\n" + "=" * 70)
print("CREATING CLEAN DATASET")
print("=" * 70)

# Create clean dataset
clean_df = df[['channel_name'] + clean_features + [target]].copy() if PLATFORM == 'youtube' else df[['username'] + clean_features + [target]].copy()

output_file = f'{PLATFORM}_features_clean.csv'
clean_df.to_csv(output_file, index=False)

print(f"\n Saved clean dataset: {output_file}")
print(f"   Samples: {len(clean_df)}")
print(f"   Features: {len(clean_features)}")
print(f"   Samples/feature: {len(clean_df)/len(clean_features):.1f}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print(f"""
 Action Items:

1. Review removed features above
   - Do they make sense to remove?
   - Any that should be kept?

2. Re-run CV analysis with clean features:
   - Use {output_file}
   - Expected R²: 0.50-0.75 (realistic!)
   - Should see NO perfect predictions

3. If R² is still too high (>0.90):
   - Further reduce features to top 5
   - Check for remaining leakage

4. Report THESE results in dissertation:
   - Be transparent about data leakage issue
   - Explain feature selection process
   - Show improved methodology

 Remember: R² = 0.60-0.70 with clean features is 
   BETTER than R² = 1.00 with leakage!
""")

print("\n Data leakage analysis complete!")