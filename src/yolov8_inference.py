import os

import yaml
from PIL import Image
from ultralytics import YOLO


def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def detect_yolov8(params):
    for version in params['versions']:
        model = YOLO(
            os.path.join(
                params['project'], f'yolov8{version}_best.pt',
            ),
        )


        image = Image.open(params['data'])

        results = model.predict(
            source=image,
            imgsz=params['imgsz'],
            save=True,
            save_txt=True,
        )



def main():
    params = parse_yaml('param_detect_YOLOv8.yaml')
    detect_yolov8(params)


if __name__ == '__main__':
    main()
