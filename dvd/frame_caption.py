import copy
import functools
import json
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

from tqdm import tqdm

import dvd.config as config
from dvd.utils import call_openai_model_with_tools

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

messages = [
    {
        "role": "system",
        "content": ""
    },
    {
        "role": "user",
        "content": "",
    },
]


CAPTION_PROMPT = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

Output template:
{
  "clip_start_time": CLIP_START_TIME,
  "clip_end_time": CLIP_END_TIME,
  "subject_registry": {
    <subject_i>: {
      "name": <fill with short identity if name is unknown>,
      "appearance": <list of appearance descriptions>,
      "identity": <list of identity descriptions>,
      "first_seen": <timestamp>
    },
    ...
  },
  "clip_description": <smooth and detailed natural narration of the video clip>
}
"""


MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video. They may contain duplicated subjects with slightly different IDs or descriptions.

Task:
1. Merge these partial registries into one coherent `subject_registry`.
2. Preserve all unique subjects.
3. If two subjects obviously refer to the same person, merge them
   (keep earliest `first_seen` time and union all fields).

Input (list of JSON objects):
REGISTRIES_PLACEHOLDER

Return *only* the merged `subject_registry` JSON object.
"""

SYSTEM_PROMPT = "You are a helpful assistant."

# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #
def convert_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def gather_frames_from_time_ranges(
    frame_folder: str, time_ranges: List[Tuple[int, int, str]]
) -> Dict[str, Dict]:
    """Return a dict keyed by 't1_t2' -> {files, transcript}."""
    frame_files = sorted(
        [f for f in os.listdir(frame_folder) if f.endswith(".jpg")],
        key=lambda x: float(x.split("_n")[-1].rstrip(".jpg")),
    )
    result = {}
    for t1, t2, text in time_ranges:
        files = frame_files[t1 : t2 + 1]
        result[f"{t1}_{t2}"] = {
            "files": [os.path.join(frame_folder, f) for f in files],
            "transcript": text or "No transcript.",
        }
    return result

def gather_clip_frames(
    video_frame_folder, clip_secs: int, subtitle_file_path: str = None
) -> Dict[str, Dict]:
    # Fix possible typo in the earlier list-comprehension and gather frames again
    frame_files = sorted(
        [f for f in os.listdir(video_frame_folder) if f.startswith("frame") and f.endswith(".jpg")],
        key=lambda x: float(x.split("_n")[-1].rstrip(".jpg")),
    )
    if not frame_files:
        return {}

    # Optional subtitle information
    subtitle_map = (
        parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}
    )

    # Map timestamps → file names for quick lookup
    frame_ts = [float(f.split("_n")[-1].rstrip(".jpg")) / config.VIDEO_FPS for f in frame_files]
    ts_to_file = dict(zip(frame_ts, frame_files))
    last_ts = int(max(frame_ts))

    result = []

    # Iterate over fixed-length clips
    clip_start = 0
    while clip_start <= last_ts:
        clip_end = min(clip_start + clip_secs - 1, last_ts)

        # Collect frames that fall inside the current clip
        clip_files = [
            os.path.join(video_frame_folder, ts_to_file[t])
            for t in frame_ts
            if clip_start <= t <= clip_end
        ]

        # Aggregate transcript text overlapping the clip interval
        transcript_parts: List[str] = []
        for key, text in subtitle_map.items():
            s, e = map(int, key.split("_"))
            if s <= clip_end and e >= clip_start:  # overlap check
                transcript_parts.append(text)
        transcript = " ".join(transcript_parts).strip() or "No transcript."

        result.append((
                f"{clip_start}_{clip_end}", 
                {"files": clip_files, "transcript": transcript}
        ))

        clip_start += clip_secs
    return result


# --------------------------------------------------------------------------- #
#                   Subtitle (.srt) parsing helper function                    #
# --------------------------------------------------------------------------- #
def _timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' to seconds (float)."""
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """
    Parse an .srt file and return a mapping
    '{startSec_endSec}': 'subtitle text'.
    """
    if not os.path.isfile(srt_path):
        return {}

    result: Dict[str, str] = {}
    with open(srt_path, "r", encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]

    idx = 0
    n = len(lines)
    while idx < n:
        # Skip sequential index if present
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break

        # Time-range line
        if "-->" not in lines[idx]:
            idx += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[idx].split("-->")]
        start_sec = int(_timestamp_to_seconds(start_ts))
        end_sec = int(_timestamp_to_seconds(end_ts))
        idx += 1

        # Collect subtitle text (may span multiple lines)
        subtitle_lines: List[str] = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        subtitle = " ".join(subtitle_lines)
        key = f"{start_sec}_{end_sec}"
        if key in result:  # append if duplicate key
            result[key] += " " + subtitle
        else:
            result[key] = subtitle
        # Skip blank line separating entries
        idx += 1
    return result


# --------------------------------------------------------------------------- #
#                        LLM wrappers (single clip)                           #
# --------------------------------------------------------------------------- #
def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder) -> Tuple[str, dict]:
    """LLM call for one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files, transcript = info["files"], info["transcript"]

    clip_start_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[1]))

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = CAPTION_PROMPT.replace(
        "TRANSCRIPT_PLACEHOLDER", transcript).replace(
        "CLIP_START_TIME", clip_start_time).replace(
        "CLIP_END_TIME", clip_end_time)

    if os.path.exists(os.path.join(caption_ckpt_folder, f"{timestamp}.json")):
        # If the caption already exists, skip processing
        with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "r") as f:
            return timestamp, json.load(f)

    tries = 3
    while tries:
        tries -= 1
        resp = call_openai_model_with_tools(
            send_messages,
            endpoints=config.AOAI_CAPTION_VLM_ENDPOINT_LIST,
            model_name=config.AOAI_CAPTION_VLM_MODEL_NAME,
            return_json=True,
            image_paths=files,
            api_key=config.OPENAI_API_KEY,
        )["content"]
        if resp is None:
            continue
        try:
            assert isinstance(resp, str), f"Response must be a JSON string instead of {type(resp)}:{resp}."
            parsed = json.loads(resp)
            parsed["clip_description"] += f"\n\nTranscript during this video clip: {transcript}." # add transcript to description
            resp = json.dumps(parsed)
            with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "w") as f:
                f.write(resp)
            return timestamp, parsed
        except json.JSONDecodeError:
            continue
    return timestamp, {}  # give up


# --------------------------------------------------------------------------- #
#                  LLM wrapper – merge subject registries                     #
# --------------------------------------------------------------------------- #
def merge_subject_registries(registries: List[dict]) -> dict:
    """Ask another LLM to merge all `new_subject_registry` dicts."""
    if not registries:
        return {}

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = MERGE_PROMPT.replace(
        "REGISTRIES_PLACEHOLDER", json.dumps(registries)
    )

    tries = 3
    while tries:
        tries -= 1
        resp = call_openai_model_with_tools(
            send_messages,
            endpoints=config.AOAI_CAPTION_VLM_ENDPOINT_LIST,
            model_name=config.AOAI_CAPTION_VLM_MODEL_NAME,
            return_json=True,
            api_key=config.OPENAI_API_KEY,
        )["content"]
        if resp is None:
            continue
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            continue
    return {}  # fallback


# --------------------------------------------------------------------------- #
#                     Process one video (parallel caption)                    #
# --------------------------------------------------------------------------- #
def process_video(
    frame_folder: str,
    output_caption_folder: str,
    subtitle_file_path: str = None,
):
    caption_ckpt_folder = os.path.join(output_caption_folder, "ckpt")
    os.makedirs(caption_ckpt_folder, exist_ok=True)

    clips = gather_clip_frames(frame_folder, config.CLIP_SECS, subtitle_file_path)

    caption_clip = functools.partial(
        _caption_clip,
        caption_ckpt_folder=caption_ckpt_folder,
    )
    # ---------------- Parallel captioning --------------- #
    with mp.Pool(16) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(caption_clip, clips),
                total=len(clips),
                desc=f"Captioning {frame_folder}",
            )
        )

    # ---------------- Save per-clip JSON ---------------- #
    partial_registries = []
    frame_captions = {}
    results = sorted(results, key=lambda x: float(x[0].split("_")[0]))
    for ts, parsed in results:
        if parsed:
            frame_captions[ts] = {
                "caption": parsed["clip_description"],
            }
            partial_registries.append(parsed["subject_registry"])

    # ---------------- Merge subject registries ---------- #
    merged_registry = merge_subject_registries(partial_registries)
    frame_captions["subject_registry"] = merged_registry

    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)


def process_video_lite(
    output_caption_folder: str,
    subtitle_file_path: str,
):
    """
    Process video in LITE_MODE using SRT subtitles.
    """
    captions = parse_srt_to_dict(subtitle_file_path)
    frame_captions = {}
    for key, text in captions.items():
        frame_captions[key] = {
            "caption": f"\n\nTranscript during this video clip: {text}.",
        }
    frame_captions["subject_registry"] = {}
    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)

# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    frame_folder = "/home/xiaoyizhang/DVD/video_database/i2qSxMVeVLI/frames"
    output_caption_folder = "/home/xiaoyizhang/DVD/video_database/i2qSxMVeVLI/captions"
    subtitle_file_path = "/home/xiaoyizhang/DVD/video_database/raw/i2qSxMVeVLI.srt"
    process_video(
        frame_folder,
        output_caption_folder,
        subtitle_file_path=subtitle_file_path,
    )

if __name__ == "__main__":
    main()