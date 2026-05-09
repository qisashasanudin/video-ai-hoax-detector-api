#!/usr/bin/env python3
"""
Test script to extract frames from the AI YouTube video and run AI detection.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from youtube_pipeline import extract_youtube_media
from ai_detection.frame_detector import detect_ai_generation_from_frames

async def test_ai_detection():
    # The YouTube video URL that scored 35%
    url = "https://www.youtube.com/shorts/szqXppELItw"

    # Create a test job ID
    job_id = "test_ai_video"

    # Base data directory
    base_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print(f"Extracting media from: {url}")

    try:
        # Extract frames and audio from the video
        extraction = await extract_youtube_media(
            url=url,
            job_id=job_id,
            base_data_dir=base_data_dir,
            max_download_seconds=60,  # Give more time for download
            max_frames=24,
        )

        print(f"Extraction successful:")
        print(f"  Job dir: {extraction.get('job_dir')}")
        print(f"  Video path: {extraction.get('video_path')}")
        print(f"  Audio path: {extraction.get('audio_path')}")
        print(f"  Frames count: {extraction.get('frames_count')}")

        frames_count = extraction.get('frames_count')
        if frames_count and frames_count > 0:
            frames_dir = os.path.join(extraction['job_dir'], 'frames')

            print(f"\nRunning AI detection on {frames_count} frames...")

            # Run AI detection
            ai_score, ai_drivers = detect_ai_generation_from_frames(
                frames_dir,
                url=url,
                max_frames=24,
            )

            print("\nAI Detection Results:")
            print(f"  AI Score: {ai_score:.3f} ({ai_score*100:.1f}%)")
            print("  AI Drivers:")
            for driver in ai_drivers:
                print(f"    - {driver}")

            # Let's also check individual frame scores
            print("\nAnalyzing individual frames...")
            from ai_detection.deepfake_detector import analyze_frame_for_ai

            frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
            for frame_path in frame_files[:5]:  # Check first 5 frames
                try:
                    frame_result = analyze_frame_for_ai(str(frame_path))
                    if isinstance(frame_result, tuple):
                        frame_score, frame_details = frame_result
                        print(f"  {frame_path.name}: {frame_score:.3f} (model: {frame_details.get('model_score', 0):.3f}, freq: {frame_details.get('frequency_score', 0):.3f})")
                    else:
                        print(f"  {frame_path.name}: {frame_result:.3f}")
                except Exception as e:
                    print(f"  {frame_path.name}: Error - {e}")

        else:
            print("No frames extracted!")

    except Exception as e:
        print(f"Error during extraction/detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ai_detection())