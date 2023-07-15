import os

import cv2
import yaml
from PIL import Image
from ultralytics import YOLO


def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def detect_yolov8(params):
    for version in params['versions']:
        # Load the model
        model = YOLO(
            os.path.join(
            params['project'], f'yolov8{version}_best.pt',
            ),
        )

        # If you want to use an image, open it with PIL or cv2
        image = Image.open(params['data'])
        # image = cv2.imread(params['data'])

        # Use the model to detect
        results = model.predict(
            source=image,
            imgsz=params['imgsz'],
            save=True,
            save_txt=True,
        )

        # Save results (if needed)
        # results.save()  # Save results (image with detections)


def main():
    params = parse_yaml('param_detect_YOLOv8.yaml')
    detect_yolov8(params)


if __name__ == '__main__':
    main()
