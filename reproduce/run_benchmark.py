import dvd.config as config
import os
import argparse
import json
from dvd.dvd_core import DVDCoreAgent
from dvd.video_utils import load_video, decode_video_to_frames, download_srt_subtitle
from dvd.frame_caption import process_video, process_video_lite
from dvd.utils import extract_answer

def main():
    parser = argparse.ArgumentParser(description="Run DVDCoreAgent on a video.")
    parser.add_argument("benchmark_database_folder", help="The path to the benchmark database folder.")
    parser.add_argument("benchmark_metadata", help="The path to the benchmark metadata file.")
    args = parser.parse_args()

    benchmark_database_folder = args.benchmark_database_folder

    with open(args.benchmark_metadata, "r") as f:
        lines = f.readlines()

    total_data = []
    results = {}
    for line in lines:
        # one line for one video instance containing multiple questions
        video_info = json.loads(line)
        video_id = video_info["key"]
        qa_list = video_info["qa"]

        qids = [qa["uid"] for qa in qa_list]
        questions = [qa["question"] for qa in qa_list]

        frames_dir = os.path.join(benchmark_database_folder, video_id, "frames")
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            print(f"Frames for video {frames_dir} not found, skipping...")
            continue
        video_db_path = os.path.join(benchmark_database_folder, video_id, "database.json")

        print(f"Initializing DVDCoreAgent from database {video_db_path}...")
        agent = DVDCoreAgent(video_db_path, video_caption_path=None, max_iterations=15)
        agent.messages[-1]['content'] += "\nSelect the best option that accurately addresses the question.\nAnswer with the option\'s letter from the given choices directly and only give the best option."
        print("Agent initialized.")
        # Run with questions
        msgs = agent.parallel_run(questions, max_workers=4)
        for qid, question, msg in zip(qids, questions, msgs):
            answer = extract_answer(msg[-1])
            results[qid] = {
                "question": question,
                "answer": answer,
                "reasoning": msg
            }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

