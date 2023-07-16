import os
import subprocess

import yaml


def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def detect_yolov7(params):
    for version in params['versions']:
        command = f"python {params['yolo_path']}/test.py --weights {os.getcwd()}/models/{version}_best_pothole.pt --img {params['img_size']} --data {os.path.join(params['data_path'], 'data_local.yaml')} --project {params['project_name']} --name {params['run_name']}_{version} --save-txt"
        subprocess.run(command, shell=True)


def main():
    params = parse_yaml('src/param_YOLOv7.yaml')
    detect_yolov7(params)


if __name__ == '__main__':
    main()
