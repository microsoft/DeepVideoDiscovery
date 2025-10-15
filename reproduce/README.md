# Reproduce

0. Setup database root path with `export DATABASE_DIR=/path/to/your/database/folder`.

1. Download the pre-built database from [here](https://huggingface.co/datasets/xyzhang626/LongVideoBenchmarkCaptions/tree/main). Or you can use the script `wget https://huggingface.co/datasets/xyzhang626/LongVideoBenchmarkCaptions/resolve/main/LVBench_4.1.zip` to download the database.

2. Prepare the database json files. You can use the script in `prepare_lvbench_db.py` to prepare the database json files. Please modify the path to your downloaded LVBench database. It will generate the database json files into `$DATABASE_DIR/LVBench_4.1`.

```bash
python -m reproduce.prepare_database /path/to/your/zipfile $DATABASE_DIR
```

3. Download LVBench dataset, you could find this 3rd party assets in [here](https://huggingface.co/datasets/AIWinter/LVBench/tree/main). Or you can use the script to download the dataset.

```bash
export TARGET_DIR=$DATABASE_DIR/LVBench_4.1
bash reproduce/download_lvbench.sh
```

4. Decode the videos into raw frames, you could use the script in `decode_frames.py`, please modify the path to your downloaded LVBench dataset.

```bash
python -m reproduce.decode_frames --part $DATABASE_DIR/LVBench_4.1/all_videos_split.zip.001 --out $TARGET_DIR --fps 2
```

5. Run the benchmark. You can use the script in `run_benchmark.py` to run the benchmark. Please modify the path to your prepared database json files.
```bash
python -m reproduce.run_benchmark $TARGET_DIR  $TARGET_DIR/video_info.meta.jsonl
```
