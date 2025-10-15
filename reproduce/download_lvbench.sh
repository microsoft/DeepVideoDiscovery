mkdir -p "$TARGET_DIR"
printf '%03d\n' {3..14} | \
wget --continue -P "$TARGET_DIR" "https://huggingface.co/datasets/zai-org/LVBench/resolve/main/video_info.meta.jsonl"
xargs -P 8 -I{} wget --continue -P "$TARGET_DIR" \
  "https://huggingface.co/datasets/AIWinter/LVBench/resolve/main/all_videos_split.zip.{}"
