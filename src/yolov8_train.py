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
            data=os.path.join(params['data_path'], 'data_local.yaml'),
            imgsz=params['img_size'],
            epochs=params['epochs'],
            batch=params['batch'],
            name=f"{params['run_name']}_{version}",
            save_period=params['save_period'],
            device=params['device'],
            project=params['project'],
            exist_ok=True,  # overwrite existing experiment
        )


def main():
    params = parse_yaml('src/param_YOLOv8.yaml')
    train_yolov8(params)


if __name__ == '__main__':
    main()
