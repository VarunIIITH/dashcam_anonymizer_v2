#!/usr/bin/env bash
# Download the two detection models into models/
set -e
mkdir -p models
# plate + face model from Dashcam Anonymizer v1 (https://github.com/varungupta31/dashcam_anonymizer)
[ -f models/best.pt ] || gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O models/best.pt
# face model from yolo-face (https://github.com/akanametov/yolo-face)
[ -f models/yolov11l-face.pt ] || curl -L -o models/yolov11l-face.pt \
  https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt
echo "Models ready in models/"
