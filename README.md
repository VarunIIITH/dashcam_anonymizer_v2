# Dashcam Anonymizer v2

Blur faces and license plates in long dashcam and egocentric videos.

v2 builds on [Dashcam Anonymizer](https://github.com/varungupta31/dashcam_anonymizer) and was used to anonymize the full-ride videos of the [MOTOR dataset](https://huggingface.co/datasets/varunpaturkar/MOTOR) (about 25 hours of front, rear, helmet and eye-tracker video).

## What's new in v2

- **Two detectors:** the v1 YOLOv8 plate + face model, plus a dedicated YOLO face model.
- **High-resolution inference** (1280 px by default) in GPU batches, so small and distant faces and plates are found.
- **Temporal smoothing:** each frame is blurred with every box detected within ±2 frames, so a single missed detection does not leave a face or plate visible.
- **Padded boxes and size-adaptive blur**; very large boxes are only blurred when the detector is confident, which removes most false positives.
- **Streams video through ffmpeg:** no extracted frames or per-frame text files on disk, optional frame-range trimming, audio removed, H.265 (NVENC) output.
- **All detections saved** to one compressed `.npz` per video.

## Installation

```bash
conda create -n anonymizer python=3.11 -y
conda activate anonymizer
pip install -r requirements.txt
bash download_models.sh
```

ffmpeg must be installed and on your `PATH`. The default encoder, `hevc_nvenc`, needs an NVIDIA GPU and an ffmpeg build with NVENC; otherwise use `--codec libx264`.

## Usage

```bash
python anonymize.py \
  --input video.mp4 --output video_blurred.mp4 \
  --plate_face_model models/best.pt --face_model models/yolov11l-face.pt
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--detections out.npz` | off | save every detection (boxes, scores, classes) |
| `--start_frame`, `--end_frame` | whole video | anonymize and keep only frames `[start, end)` |
| `--imgsz` | 1280 | inference resolution |
| `--conf`, `--face_conf` | 0.1, 0.15 | confidence thresholds of the plate+face model and the face model |
| `--window` | 2 | blur each frame with boxes from ±`window` frames |
| `--pad` | 0.15 | enlarge each box by this fraction on every side |
| `--big_plate_area`, `--big_face_area`, `--big_conf` | 0.03, 0.10, 0.5 | boxes larger than this fraction of the frame need confidence ≥ `big_conf` |
| `--codec`, `--cq` | `hevc_nvenc`, 28 | output encoder and quality (lower `--cq` = higher quality) |
| `--out_rate` | input rate | output frame rate, e.g. `30` |
| `--device`, `--batch` | `0`, 8 | GPU id (or `cpu`) and batch size |

Run `python anonymize.py --help` for the full list.

## Tested with

Python 3.11, ultralytics 8.4, PyTorch 2.4 (CUDA 11.8), OpenCV 4.11, ffmpeg 4.4, on NVIDIA GTX 1080 Ti GPUs.

## Acknowledgements

- [Dashcam Anonymizer](https://github.com/varungupta31/dashcam_anonymizer) (v1) by Varun Gupta, whose plate + face model v2 uses.
- Face model from [yolo-face](https://github.com/akanametov/yolo-face).
- Detection with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

The models and libraries above are distributed under their own licenses.

## Citation

If this repository helped you in a research project, please consider citing it:

```bibtex
@software{dashcam_anonymizer_v2,
  author = {Varun Paturkar},
  month = {10},
  title = {{Dashcam Anonymizer v2}},
  url = {https://github.com/VarunIIITH/dashcam_anonymizer_v2},
  version = {2.0.0},
  year = {2026}
}
```

## License

MIT (see `LICENSE`).
