"""
ALTERNATIVE METHOD: Discover channels through trending videos
This finds channels by looking at who created popular videos, then checks their subscriber count
"""

from googleapiclient.discovery import build
import pandas as pd
import time

# ========================================
# PUT YOUR API KEY HERE
# ========================================
API_KEY = 'AIzaSyBshx6Lsbs6rpnt0NbT9c034xSDLnTccS8'

# YouTube Video Categories
CATEGORIES = {
    '22': 'People & Blogs',
    '26': 'Howto & Style',
    '24': 'Entertainment',
    '28': 'Science & Technology',
    '20': 'Gaming',
    '27': 'Education',
    '23': 'Comedy',
    '17': 'Sports',
    '25': 'News & Politics',
}

print("=" * 70)
print("CHANNEL DISCOVERY via TRENDING VIDEOS")
print("=" * 70)
print("\nStrategy: Find popular videos, check their creators' subscriber counts\n")

youtube = build('youtube', 'v3', developerKey=API_KEY)

found_channels = {}  # Use dict to avoid duplicates
target_channels = 40

for category_id, category_name in CATEGORIES.items():
    if len(found_channels) >= target_channels:
        break
    
    print(f"\n Searching {category_name} (Category {category_id})...")
    
    try:
        # Search for popular videos in this category
        search_request = youtube.search().list(
            part='snippet',
            type='video',
            videoCategoryId=category_id,
            maxResults=20,
            order='viewCount',  # Most viewed videos
            relevanceLanguage='en'
        )
        search_response = search_request.execute()
        
        # Extract unique channel IDs from these videos
        channel_ids = set()
        for item in search_response.get('items', []):
            channel_id = item['snippet']['channelId']
            channel_ids.add(channel_id)
        
        print(f"  Found {len(channel_ids)} unique channels to check...")
        
        # Now check each channel's subscriber count
        for channel_id in channel_ids:
            if channel_id in found_channels:
                continue  # Already checked this channel
            
            if len(found_channels) >= target_channels:
                break
            
            try:
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
                
                # Check subscriber count
                if 'subscriberCount' not in stats:
                    continue
                
                sub_count = int(stats['subscriberCount'])
                
                # Target range: 100K to 1M
                if 100000 <= sub_count <= 1000000:
                    found_channels[channel_id] = {
                        'channel_id': channel_id,
                        'channel_name': snippet['title'],
                        'channel_url': f'https://youtube.com/channel/{channel_id}',
                        'subscribers': sub_count,
                        'video_count': int(stats['videoCount']),
                        'total_views': int(stats['viewCount']),
                        'created_at': snippet['publishedAt'],
                        'country': snippet.get('country', 'Unknown'),
                        'category': category_name,
                        'custom_url': snippet.get('customUrl', '')
                    }
                    print(f"   {snippet['title'][:40]:40} | {sub_count:,} subs")
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"    Error checking channel: {e}")
                continue
        
        print(f"  Total found so far: {len(found_channels)}/{target_channels}")
        time.sleep(2)  # Rate limiting between categories
        
    except Exception as e:
        print(f"   Error searching category: {e}")
        time.sleep(2)

# Convert to DataFrame and save
if len(found_channels) > 0:
    df = pd.DataFrame(list(found_channels.values()))
    df = df.sort_values('subscribers', ascending=False)
    df.to_csv('channels_from_trending.csv', index=False)
    
    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE!")
    print("=" * 70)
    print(f"\n RESULTS:")
    print(f"  Channels found: {len(df)}")
    print(f"  Categories searched: {len(CATEGORIES)}")
    
    print(f"\n FILE SAVED:")
    print(f"  - channels_from_trending.csv")
    
    print(f"\n STATISTICS:")
    print(f"  Subscriber range: {df['subscribers'].min():,} - {df['subscribers'].max():,}")
    print(f"  Average: {df['subscribers'].mean():,.0f}")
    
    print(f"\n CATEGORIES REPRESENTED:")
    print(df['category'].value_counts().to_string())
    
    print(f"\n TOP 10 CHANNELS:")
    for idx, row in df.head(10).iterrows():
        print(f"  {row['channel_name'][:45]:45} | {row['subscribers']:,} | {row['category']}")
    
    if len(df) >= 40:
        print(f"\n SUCCESS! You have enough channels for your research!")
    else:
        print(f"\n  Found {len(df)}/40 channels. Run discover_channels_active.py to find more!")
    
else:
    print("\n No channels found. Check API key and quota.")