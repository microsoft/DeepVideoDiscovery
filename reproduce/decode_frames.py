import argparse
import os
import re
import sys
import tempfile
import zipfile
import shutil
import logging
from pathlib import Path
from typing import List, Iterable
import cv2
from tqdm import tqdm
import multiprocessing as mp

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpg', '.mpeg', '.m4v'}


def find_all_parts(part_path: Path) -> List[Path]:
    """
    Given any split archive file, find all parts in the same directory and return them in order.
    Supports *.zip.001 / *.zip.002 or *.z01 / *.z02 + .zip formats (compatible with common naming).
    """
    name = part_path.name
    parent = part_path.parent

    # Match something like something.zip.001 or something.zip.002
    m = re.match(r'(.+\.zip)\.(\d{3})$', name)
    if m:
        base = m.group(1)
        parts = sorted(parent.glob(base + ".???"))
        return parts

    # Match WinZip style: something.z01, something.z02 ... + something.zip
    m2 = re.match(r'(.+)\.z(\d{2})$', name, re.IGNORECASE)
    if m2:
        base_prefix = m2.group(1)
        zparts = sorted(parent.glob(base_prefix + ".z??"), key=lambda p: p.suffix.lower())
        main_zip = parent / (base_prefix + ".zip")
        if main_zip.exists():
            return zparts + [main_zip]

    # If it's a complete zip file
    if name.endswith(".zip"):
        return [part_path]

    raise ValueError(f"Unrecognized split archive naming: {name}")


def assemble_zip(parts: List[Path]) -> Path:
    """
    Concatenate all parts in order into a temporary zip file and return its path.
    If there is only one .zip file, return its path directly (no copy).
    """
    if len(parts) == 1 and parts[0].suffix == ".zip":
        return parts[0]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="merged_zip_")
    os.close(tmp_fd)
    with open(tmp_path, "wb") as w:
        for p in parts:
            logging.info(f"Merging part: {p.name}")
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, length=1024 * 1024)
    return Path(tmp_path)


def iter_video_members(zf: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in zf.infolist():
        if info.is_dir():
            continue
        ext = Path(info.filename).suffix.lower()
        if ext in VIDEO_EXTS:
            yield info


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def decode_video(temp_video_path: Path, out_root: Path, fps: float, overwrite: bool = False, video_stem: str | None = None):
    """
    Extract frames at the given fps and save them.
    video_stem: Pass the original video filename (without extension) to avoid using the temporary filename.
    """
    cap = cv2.VideoCapture(str(temp_video_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video: {temp_video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if orig_fps <= 0:
        logging.warning(f"Original FPS abnormal ({orig_fps}), will estimate by frame time.")
        orig_fps = fps  # fallback

    interval = 1.0 / fps
    next_t = 0.0
    frame_index = 0
    saved_index = 0

    # Use original filename (if provided)
    video_stem = video_stem or temp_video_path.stem
    frames_dir = out_root / video_stem / "frames"
    ensure_dir(frames_dir)

    if not overwrite:
        # Count existing frames, auto-continue
        existing = sorted(frames_dir.glob("frames_n*.jpg"))
        if existing:
            last = existing[-1].stem
            m = re.search(r'frames_n(\d+)', last)
            if m:
                saved_index = int(m.group(1))
                logging.info(f"{video_stem}: Append mode, {saved_index} frames already exist.")
    
    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                desc=f"Decoding {video_stem}",
                unit="f",
                dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_t = frame_index / orig_fps
        if current_t + 1e-6 >= next_t:
            saved_index += 1
            out_path = frames_dir / f"frames_n{saved_index:06d}.jpg"
            if overwrite or not out_path.exists():
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            next_t += interval
        frame_index += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    logging.info(f"{video_stem}: Frame extraction complete, total (including existing) {saved_index} frames.")


def process_archive(part_path: Path, out_root: Path, fps: float, overwrite: bool):
    parts = find_all_parts(part_path)
    logging.info("Found parts in order: " + ", ".join(p.name for p in parts))
    merged_zip = assemble_zip(parts)
    cleanup_needed = merged_zip not in parts  # If we generated a temporary file
    try:
        with zipfile.ZipFile(merged_zip, 'r') as zf:
            video_members = list(iter_video_members(zf))
            logging.info(f"Number of video files in archive: {len(video_members)}")
            temp_files = []  # (process, tmp_path)
            max_workers = min(len(video_members), mp.cpu_count() or 1)
            logging.info(f"Parallel decoding: using {max_workers} processes")

            def wait_one():
                # Wait for the earliest started process to finish and clean up temp file
                proc, tpath = temp_files.pop(0)
                proc.join()
                try:
                    tpath.unlink(missing_ok=True)
                except Exception:
                    pass

            for info in video_members:
                # Write this video to a temporary file
                suffix = Path(info.filename).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(zf.read(info))
                    tmp_path = Path(tmp.name)

                original_stem = Path(info.filename).stem
                # Start a separate process for decoding, passing the original filename
                p = mp.Process(target=decode_video, args=(tmp_path, out_root, fps), kwargs={'overwrite': overwrite, 'video_stem': original_stem})
                p.start()
                temp_files.append((p, tmp_path))

                # If reached parallel limit, wait for the earliest one to finish
                if len(temp_files) >= max_workers:
                    wait_one()

            # Wait for all remaining to finish
            while temp_files:
                wait_one()
    finally:
        if cleanup_needed:
            try:
                merged_zip.unlink(missing_ok=True)
            except Exception:
                pass


def parse_args():
    ap = argparse.ArgumentParser(description="Extract video frames from split zip archives at a given fps")
    ap.add_argument("--part", required=True, help="Path to any split archive file (e.g. all_videos_split.zip.002)")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--fps", type=float, required=True, help="Target frame extraction rate (e.g. 5)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    part_path = Path(args.part).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    ensure_dir(out_root)

    if args.fps <= 0:
        logging.error("fps must be > 0")
        sys.exit(1)

    if not part_path.exists():
        logging.error(f"File does not exist: {part_path}")
        sys.exit(1)

    process_archive(part_path, out_root, args.fps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()