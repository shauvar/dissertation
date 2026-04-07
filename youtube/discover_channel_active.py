"""
ACTIVE YouTube Channel Discovery Script
Searches YouTube and automatically filters by current subscriber count (100K-1M)
"""

from googleapiclient.discovery import build
import pandas as pd
import time
import random

# ========================================
# PUT YOUR API KEY HERE
# ========================================
API_KEY = 'AIzaSyBshx6Lsbs6rpnt0NbT9c034xSDLnTccS8'

# ========================================
# SEARCH TERMS - Feel free to modify!
# ========================================
SEARCH_TERMS = [
    # Fashion & Beauty
    'fashion lookbook', 'makeup routine', 'style guide', 'thrift haul',
    'beauty tips', 'outfit ideas', 'skincare review',
    
    # Tech & Gaming
    'tech tutorial', 'coding tutorial', 'game walkthrough', 'tech review',
    'programming guide', 'tech explained', 'gaming tips',
    
    # Lifestyle
    'productivity vlog', 'morning routine', 'day in life', 'minimalist living',
    'home organization', 'student life', 'lifestyle tips',
    
    # Food
    'easy recipes', 'cooking tutorial', 'meal prep', 'baking guide',
    'food review', 'recipe video', 'cooking tips',
    
    # Fitness
    'home workout', 'fitness routine', 'gym tips', 'yoga flow',
    'workout guide', 'fitness motivation',
    
    # Education
    'explained simply', 'tutorial guide', 'how to learn', 'study tips',
    'educational video', 'lessons',
    
    # Entertainment
    'comedy sketch', 'funny video', 'reaction video', 'music cover',
    'animation', 'short film',
]

print("=" * 70)
print("YOUTUBE CHANNEL DISCOVERY - 100K to 1M Subscribers")
print("=" * 70)
print(f"\nSearching with {len(SEARCH_TERMS)} different topics...")
print("Target: 40 channels in range\n")

# Initialize
youtube = build('youtube', 'v3', developerKey=API_KEY)

found_channels = []
seen_channel_ids = set()
search_attempts = 0

for search_term in SEARCH_TERMS:
    if len(found_channels) >= 40:
        break
    
    search_attempts += 1
    print(f"[Search {search_attempts}] Topic: '{search_term}'")
    
    try:
        # Search for channels (not videos) related to this topic
        search_request = youtube.search().list(
            part='snippet',
            q=search_term,
            type='channel',
            maxResults=10,  # Get 10 channels per search
            order='relevance'
        )
        search_response = search_request.execute()
        
        channels_checked = 0
        channels_found_this_search = 0
        
        for item in search_response.get('items', []):
            channel_id = item['id']['channelId']
            
            # Skip if we've already checked this channel
            if channel_id in seen_channel_ids:
                continue
            
            seen_channel_ids.add(channel_id)
            channels_checked += 1
            
            # Get detailed channel info
            channel_request = youtube.channels().list(
                part='statistics,snippet',
                id=channel_id
            )
            channel_response = channel_request.execute()
            
            if not channel_response['items']:
                continue
            
            channel_data = channel_response['items'][0]
            stats = channel_data['statistics']
            snippet = channel_data['snippet']
            
            # Check if subscriber count is available and in range
            if 'subscriberCount' not in stats:
                continue
            
            sub_count = int(stats['subscriberCount'])
            
            # Filter: 100K to 1M
            if 100000 <= sub_count <= 1000000:
                found_channels.append({
                    'channel_id': channel_id,
                    'channel_name': snippet['title'],
                    'channel_url': f'https://youtube.com/channel/{channel_id}',
                    'subscribers': sub_count,
                    'video_count': int(stats['videoCount']),
                    'total_views': int(stats['viewCount']),
                    'created_at': snippet['publishedAt'],
                    'country': snippet.get('country', 'Unknown'),
                    'description': snippet.get('description', '')[:200],
                    'search_term': search_term
                })
                
                channels_found_this_search += 1
                print(f"   Found: {snippet['title'][:40]:40} | {sub_count:,} subs")
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"  Checked: {channels_checked} | Found in range: {channels_found_this_search} | Total so far: {len(found_channels)}")
        time.sleep(1)
        
    except Exception as e:
        print(f"   Error: {e}")
        time.sleep(2)

# Save results
if len(found_channels) > 0:
    df = pd.DataFrame(found_channels)
    
    # Remove duplicates (just in case)
    df = df.drop_duplicates(subset=['channel_id'])
    
    # Sort by subscriber count
    df = df.sort_values('subscribers', ascending=False)
    
    # Save to CSV
    df.to_csv('discovered_channels_100k_1m.csv', index=False)
    
    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE!")
    print("=" * 70)
    print(f"\n SUMMARY:")
    print(f"  Search terms used: {search_attempts}")
    print(f"  Unique channels checked: {len(seen_channel_ids)}")
    print(f"  Channels found in range: {len(df)}")
    
    print(f"\n FILE SAVED:")
    print(f"  - discovered_channels_100k_1m.csv ({len(df)} channels)")
    
    print(f"\n SUBSCRIBER DISTRIBUTION:")
    print(f"  Min: {df['subscribers'].min():,}")
    print(f"  Max: {df['subscribers'].max():,}")
    print(f"  Mean: {df['subscribers'].mean():,.0f}")
    print(f"  Median: {df['subscribers'].median():,.0f}")
    
    print(f"\n SAMPLE CHANNELS FOUND:")
    for idx, row in df.head(10).iterrows():
        print(f"  • {row['channel_name'][:45]:45} | {row['subscribers']:,} subs")
    
    if len(df) < 40:
        print(f"\n  WARNING: Only found {len(df)} channels (target: 40)")
        print("Solutions:")
        print("  1. Run the script again (it searches different results each time)")
        print("  2. Add more search terms to SEARCH_TERMS list")
        print("  3. Use the manual discovery methods I provided earlier")
    else:
        print(f"\n SUCCESS! You have {len(df)} channels ready for data collection!")
        print(f" Next step: Run collect_video_data_fixed.py")
else:
    print("\n No channels found in the target range.")
    print("\nTroubleshooting:")
    print("  1. Check your API key is correct")
    print("  2. Verify you have API quota remaining (check Google Cloud Console)")
    print("  3. Try running again with different search terms")