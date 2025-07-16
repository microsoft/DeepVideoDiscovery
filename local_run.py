import dvd.config as config
import os
import argparse
from dvd.dvd_core import DVDCoreAgent
from dvd.video_utils import load_video, decode_video_to_frames, download_srt_subtitle
from dvd.frame_caption import process_video, process_video_lite
from dvd.utils import extract_answer

def main():
    parser = argparse.ArgumentParser(description="Run DVDCoreAgent on a video.")
    parser.add_argument("video_url", help="The URL of the video to process.")
    parser.add_argument("question", help="The question to ask about the video.")
    args = parser.parse_args()

    video_url = args.video_url
    question = args.question

    # Extract video ID from URL
    if "v=" in video_url:
        video_id = video_url.split("v=")[1]
    else:
        # Handle other URL formats if necessary
        video_id = os.path.splitext(os.path.basename(video_url))[0]


    video_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "frames")
    captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "database.json")
    srt_path = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "subtitles.srt")

    if config.LITE_MODE:
        print("Running in LITE_MODE.")
        if not os.path.exists(srt_path):
            print(f"Downloading SRT subtitle for {video_url} to {srt_path}...")
            try:
                download_srt_subtitle(video_url, srt_path)
                print("SRT subtitle downloaded.")
            except Exception as e:
                print(f"Error downloading subtitle: {e}")
                print("Please turn off LITE_MODE and try again.")
                return
        else:
            print(f"SRT subtitle already exists at {srt_path}.")
        
        # In LITE_MODE, we use srt as caption file
        process_video_lite(captions_dir, srt_path)
        caption_file = os.path.join(captions_dir, "captions.json")
    else:
        # Download video
        if not os.path.exists(video_path):
            print(f"Downloading video from {video_url} to {video_path}...")
            load_video(video_url, video_path)
            print("Video downloaded.")
        else:
            print(f"Video already exists at {video_path}.")

        # Decode video to frames
        if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
            print(f"Decoding video to frames in {frames_dir}...")
            decode_video_to_frames(video_path)
            print("Video decoded.")
        else:
            print(f"Frames already exist in {frames_dir}.")

        # Get captions
        caption_file = os.path.join(captions_dir, "captions.json")
        if not os.path.exists(caption_file):
            print("Processing video to get captions...")
            process_video(frames_dir, captions_dir)
            print("Captions generated.")
        else:
            print(f"Captions already exist at {caption_file}.")

    # Initialize DVDCoreAgent
    print("Initializing DVDCoreAgent...")
    agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
    print("Agent initialized.")

    # Run with question
    print(f"Running agent with question: '{question}'")
    msgs = agent.run(question)
    print(extract_answer(msgs[-1]))

if __name__ == "__main__":
    main()

