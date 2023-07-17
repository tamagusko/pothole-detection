import os
import subprocess

import yaml


def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def train_yolov5(params):
    for version in params['versions']:
        modified_run_name = f'{params["run_name"]}_YOLOv5{version}'
        command = f"python {os.path.join(params['yolo_path'], 'train.py')} --img {params['img_size']} --batch {params['batch']} --epochs {params['epochs']} --data {os.path.join(params['data_path'], 'data_local.yaml')} --weights yolov5{version}.pt --project {params['project_name']} --name {modified_run_name} --exist-ok"
        subprocess.run(command, shell=True)

        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
        os.system(
            f'mv {os.getcwd()}/yolov5{version}.pt {os.getcwd()}/models/yolov5{version}.pt',
        )
        os.system(
            f'cp {params["project_name"]}/{modified_run_name}/weights/best.pt {os.getcwd()}/models/yolov5{version}_best_pothole.pt',
        )


def main():
    params = parse_yaml('src/param_YOLOv5.yaml')
    train_yolov5(params)


if __name__ == '__main__':
    main()
