#!/usr/bin/env python3
"""Dashcam Anonymizer v2 — blur faces and license plates in long videos.

Changes from v1 (https://github.com/varungupta31/dashcam_anonymizer):
  * Two detectors: the v1 YOLOv8 plate+face model, plus a dedicated YOLO face model.
  * High-resolution inference (imgsz 1280 by default) in GPU batches.
  * Temporal smoothing: a frame is blurred with every box detected within +-`window` frames,
    so single-frame misses do not leave a face or plate visible.
  * Boxes are enlarged by `pad` and the blur strength scales with box size; oversized boxes need high confidence.
  * Streams frames through ffmpeg (optional frame-range trim, no audio, NVENC H.265 encode);
    no per-frame text files. Detections are saved to one compressed .npz per video.
"""
import argparse, collections, json, os, subprocess, sys, time
import numpy as np, cv2
from ultralytics import YOLO

PLATE, FACE = 0, 1  # output class ids
# ffmpeg's default YUV<->BGR conversion darkens the video by about 2 levels and caps a lossless round trip at 45 dB.
# These flags remove the shift (under 0.04 levels, 57 dB) and give the same result on any machine.
SWS_FLAGS = 'bicubic+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact'


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)['streams'][0]
    num, den = map(int, s['r_frame_rate'].split('/'))
    return int(s['width']), int(s['height']), num / den, s['r_frame_rate'], int(s.get('nb_frames') or 0)


def reader(path, w, h, fps, start=None, end=None):
    """Yield BGR frames; with start/end, decode only frames [start, end) (accurate seek on constant-rate video)."""
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', f'{start / fps:.6f}']
    cmd += ['-i', path]
    if end is not None:
        cmd += ['-frames:v', str(end - (start or 0))]
    cmd += ['-sws_flags', SWS_FLAGS, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
    p.wait()


def writer(path, w, h, rate, codec, cq):
    enc = {'h264_nvenc': ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', str(cq), '-b:v', '0'],
           'hevc_nvenc': ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', str(cq), '-b:v', '0', '-tag:v', 'hvc1'],
           'libx264': ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(cq)]}[codec]
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', rate,
           '-i', '-', '-sws_flags', SWS_FLAGS] + enc + ['-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class Detector:
    def __init__(self, plate_face_model, face_model, imgsz, conf, face_conf, device, batch,
                 big_plate_area=0.03, big_face_area=0.10, big_conf=0.5):
        self.big_plate_area, self.big_face_area, self.big_conf = big_plate_area, big_face_area, big_conf
        self.dropped = 0
        self.pf = YOLO(plate_face_model)
        self.f = YOLO(face_model) if face_model else None
        self.imgsz, self.conf, self.face_conf, self.device, self.batch = imgsz, conf, face_conf, device, batch
        # v1 model names: {0: 'license', 1: 'face'}
        self.pf_map = {i: (PLATE if 'lic' in n.lower() or 'plate' in n.lower() else FACE) for i, n in self.pf.names.items()}

    def __call__(self, frames):
        """Return per-frame arrays of [x1, y1, x2, y2, score, cls, source] (source 0 = v1 model, 1 = face model)."""
        out = [[] for _ in frames]
        for src, model, conf in [(0, self.pf, self.conf), (1, self.f, self.face_conf)]:
            if model is None:
                continue
            res = model.predict(frames, imgsz=self.imgsz, conf=conf, device=self.device, verbose=False, batch=self.batch)
            for i, r in enumerate(res):
                b = r.boxes
                if len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy(); sc = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy().astype(int)
                cls = np.array([self.pf_map[c] for c in cl]) if src == 0 else np.full(len(cl), FACE)
                out[i].append(np.column_stack([xyxy, sc, cls, np.full(len(cl), src)]))
        res = []
        for f, o in zip(frames, out):
            b = np.vstack(o) if o else np.zeros((0, 7))
            if len(b):
                # oversized boxes are kept only when confident (large low-confidence boxes are almost always false positives)
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (f.shape[0] * f.shape[1])
                big = ((b[:, 5] == PLATE) & (area > self.big_plate_area)) | ((b[:, 5] == FACE) & (area > self.big_face_area))
                drop = big & (b[:, 4] < self.big_conf)
                self.dropped += int(drop.sum()); b = b[~drop]
            res.append(b)
        return res


def blur_boxes(img, boxes, pad, min_kernel):
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
        y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
        if x2 <= x1 or y2 <= y1:
            continue
        k = max(min_kernel, int(0.5 * min(x2 - x1, y2 - y1)) | 1)  # odd kernel that grows with the box
        roi = img[y1:y2, x1:x2]
        # downscale -> blur -> upscale: strong, cheap blur even for large boxes
        s = max(1, k // 15)
        small = cv2.resize(roi, (max(1, roi.shape[1] // s), max(1, roi.shape[0] // s)), interpolation=cv2.INTER_AREA)
        kk = max(3, (k // s) | 1)
        small = cv2.GaussianBlur(small, (kk, kk), 0)
        img[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img


def anonymize(args):
    w, h, fps, rate, nb = probe(args.input)
    det = Detector(args.plate_face_model, args.face_model, args.imgsz, args.conf, args.face_conf, args.device, args.batch,
                   args.big_plate_area, args.big_face_area, args.big_conf)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    enc = writer(args.output, w, h, args.out_rate or rate, args.codec, args.cq)
    W = args.window
    src = reader(args.input, w, h, fps, args.start_frame, args.end_frame)
    all_dets, written, t0 = [], 0, time.time()
    hist = []                                  # detections of the last W frames already written
    pending_frames, pending_dets, batch = [], [], []

    def write_ready(final=False):
        """Write frames whose +W lookahead detections are known; each frame is blurred with boxes from +-W frames."""
        nonlocal written, hist
        while pending_frames and (final or len(pending_dets) >= W + 1):
            img = pending_frames.pop(0)
            near = hist[-W:] + pending_dets[:W + 1] if W > 0 else pending_dets[:1]
            boxes = np.vstack(near) if near else np.zeros((0, 7))
            blur_boxes(img, boxes[:, :4], args.pad, args.min_kernel)
            enc.stdin.write(img.tobytes()); written += 1
            hist.append(pending_dets.pop(0)); hist = hist[-max(W, 1):]

    for frame in src:
        batch.append(frame)
        if len(batch) == args.batch:
            d = det(batch); all_dets.extend(d)
            pending_frames.extend(batch); pending_dets.extend(d); batch = []
            write_ready()
    if batch:
        d = det(batch); all_dets.extend(d); pending_frames.extend(batch); pending_dets.extend(d)
    write_ready(final=True)
    enc.stdin.close(); enc.wait()

    if args.detections:
        np.savez_compressed(args.detections,
                            frame=np.concatenate([np.full(len(d), i) for i, d in enumerate(all_dets)]) if all_dets else np.zeros(0),
                            boxes=np.vstack(all_dets) if all_dets else np.zeros((0, 7)),
                            columns=np.array(['x1', 'y1', 'x2', 'y2', 'score', 'class(0=plate,1=face)', 'source(0=v1,1=face_model)']),
                            params=json.dumps(vars(args)))
    dt = time.time() - t0
    if args.end_frame is not None and written != args.end_frame - (args.start_frame or 0):
        print(f'WARNING: wrote {written} frames, expected {args.end_frame - (args.start_frame or 0)}', file=sys.stderr, flush=True)
    print(f'{os.path.basename(args.input)}: {written} frames in {dt:.1f}s ({written / max(dt, 1e-9):.1f} fps), '
          f'{sum(len(d) for d in all_dets)} detections, {det.dropped} oversized low-confidence boxes skipped', flush=True)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--detections', help='optional .npz path to save all detections')
    p.add_argument('--plate_face_model', required=True, help='v1 YOLOv8 model (classes: license, face)')
    p.add_argument('--face_model', default=None, help='dedicated YOLO face model (optional)')
    p.add_argument('--imgsz', type=int, default=1280); p.add_argument('--conf', type=float, default=0.1)
    p.add_argument('--face_conf', type=float, default=0.15)
    p.add_argument('--window', type=int, default=2, help='temporal window (+-frames) for box carry-over')
    p.add_argument('--pad', type=float, default=0.15, help='fractional box enlargement on each side')
    p.add_argument('--min_kernel', type=int, default=31)
    p.add_argument('--start_frame', type=int, default=None); p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--big_plate_area', type=float, default=0.03, help='plates larger than this frame fraction need --big_conf')
    p.add_argument('--big_face_area', type=float, default=0.10, help='faces larger than this frame fraction need --big_conf')
    p.add_argument('--big_conf', type=float, default=0.5)
    p.add_argument('--codec', default='hevc_nvenc', choices=['h264_nvenc', 'hevc_nvenc', 'libx264'])
    p.add_argument('--cq', type=int, default=28); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='0')
    p.add_argument('--out_rate', default=None, help='output frame rate (e.g. 30); default keeps the input rate')
    anonymize(p.parse_args())


if __name__ == '__main__':
    main()
