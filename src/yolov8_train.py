import os

import yaml
from ultralytics import YOLO


def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def train_yolov8(params):
    for version in params['versions']:
        # Load a model
        model = YOLO(f'yolov8{version}.pt')  # load a pretrained model

        # Train the model
        results = model.train(
            data=params['data'],
            epochs=params['epochs'],
            batch_size=params['batch'],
            imgsz=params['imgsz'],
            save_period=params['save_period'],
            cache=params['cache'],
            device=params['device'],
        )

        # Save model weights
        os.makedirs(
            os.path.join(
            os.getcwd(), params['project'],
            ), exist_ok=True,
        )
        results.model.save(
            os.path.join(
            params['project'], f'yolov8{version}.pt',
            ),
        )


def main():
    params = parse_yaml('param_train_YOLOv8.yaml')
    train_yolov8(params)


if __name__ == '__main__':
    main()
