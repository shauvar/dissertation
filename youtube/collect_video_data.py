"""
Corrected: Collect Video Data Script
Gets detailed metrics for each channel's recent videos
"""

from googleapiclient.discovery import build
import pandas as pd
import time
from datetime import datetime

# ========================================
# PUT YOUR API KEY HERE
# ========================================
API_KEY = 'AIzaSyBshx6Lsbs6rpnt0NbT9c034xSDLnTccS8'  # Replace with your actual API key

print("=" * 60)
print("VIDEO DATA COLLECTION")
print("=" * 60)

# Read the channels you just collected
try:
    channels_df = pd.read_csv('discovered_channels_100k_1m.csv')
    print(f"\n Loaded {len(channels_df)} channels from starter_channels_100k_1m.csv")
except FileNotFoundError:
    print("\n Error: starter_channels_100k_1m.csv not found!")
    print("Please run collect_starter_channels_fixed.py first.")
    exit(1)

# Initialize YouTube API client (CORRECTED: use developerKey)
youtube = build('youtube', 'v3', developerKey=API_KEY)

all_videos = []
failed_channels = []

print(f"\nCollecting last 50 videos from each channel...")
print("This will take about 5-10 minutes...\n")

for idx, row in channels_df.iterrows():
    channel_id = row['channel_id']
    channel_name = row['channel_name']
    
    print(f"[{idx+1}/{len(channels_df)}] Processing: {channel_name}")
    
    try:
        # Step 1: Get uploads playlist ID
        ch_request = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        )
        ch_response = ch_request.execute()
        
        if not ch_response['items']:
            print(f"   Channel not found")
            failed_channels.append(channel_name)
            continue
        
        playlist_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Step 2: Get last 50 videos from uploads playlist
        pl_request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50
        )
        pl_response = pl_request.execute()
        
        video_ids = [item['contentDetails']['videoId'] for item in pl_response['items']]
        
        if not video_ids:
            print(f"    No videos found")
            continue
        
        # Step 3: Get detailed statistics for these videos
        # YouTube API allows max 50 video IDs per request
        vid_request = youtube.videos().list(
            part='statistics,snippet,contentDetails',
            id=','.join(video_ids)
        )
        vid_response = vid_request.execute()
        
        # Step 4: Extract video data
        videos_collected = 0
        for video in vid_response['items']:
            stats = video['statistics']
            snippet = video['snippet']
            content = video['contentDetails']
            
            all_videos.append({
                'channel_id': channel_id,
                'channel_name': channel_name,
                'video_id': video['id'],
                'video_url': f"https://youtube.com/watch?v={video['id']}",
                'title': snippet['title'],
                'published_at': snippet['publishedAt'],
                'duration': content['duration'],
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'tags_count': len(snippet.get('tags', [])),
                'category_id': snippet['categoryId']
            })
            videos_collected += 1
        
        print(f"   Collected {videos_collected} videos")
        time.sleep(2)  # Be nice to API (rate limiting)
        
    except Exception as e:
        print(f"   Error: {e}")
        failed_channels.append(channel_name)
        time.sleep(2)

# Save results
videos_df = pd.DataFrame(all_videos)

if len(videos_df) > 0:
    videos_df.to_csv('youtube_videos_raw.csv', index=False)
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"\n SUMMARY:")
    print(f"  Channels processed: {len(channels_df)}")
    print(f"  Total videos collected: {len(videos_df)}")
    print(f"  Average videos per channel: {len(videos_df)/len(channels_df):.1f}")
    
    if failed_channels:
        print(f"  Failed channels: {len(failed_channels)}")
        print(f"    {', '.join(failed_channels)}")
    
    print(f"\n FILE SAVED:")
    print(f"  - youtube_videos_raw.csv ({len(videos_df)} videos)")
    
    # Calculate some basic stats
    print(f"\n VIDEO STATISTICS:")
    print(f"  Total views: {videos_df['views'].sum():,}")
    print(f"  Average views per video: {videos_df['views'].mean():,.0f}")
    print(f"  Average likes per video: {videos_df['likes'].mean():,.0f}")
    print(f"  Average comments per video: {videos_df['comments'].mean():,.0f}")
    
    print(f"\n Next step: Run calculate_features.py to compute engagement metrics!")
else:
    print("\n No videos were collected. Please check your API key and try again.")