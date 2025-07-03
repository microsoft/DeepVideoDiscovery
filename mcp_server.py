import os
from mcp.server.fastmcp import FastMCP
import dvd.config as config
from dvd.dvd_core import DVDCoreAgent
from dvd.utils import extract_answer
from dvd.video_utils import load_video, decode_video_to_frames
from dvd.frame_caption import process_video

# Initialize FastMCP server
mcp = FastMCP("dvd_agent")

def get_video_id(video_url: str) -> str:
    """Extracts video ID from URL."""
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = os.path.splitext(os.path.basename(video_url))[0]
    return video_id

@mcp.tool()
def query_video(video_url: str, question: str) -> str:
    """
    Process a video from a URL and answer a question about it.

    This tool will download the video, decode it into frames, generate captions,
    and then use the DVDCoreAgent to answer a question about the video.

    Args:
        video_url: The URL of the video to process.
        question: The question to ask about the video.
    
    Returns:
        The answer to the question from the DVDCoreAgent.
    """
    video_id = get_video_id(video_url)

    video_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "frames")
    captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "database.json")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)

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
    return extract_answer(msgs[-1])

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
