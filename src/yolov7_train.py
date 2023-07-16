import os
import subprocess

import yaml


def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def train_yolov7(params):
    for version in params['versions']:
        modified_run_name = f"{params['run_name']}_{version}"
        command = f"python {params['yolo_path']}/train.py --img-size {params['img_size']} --cfg {params['cfg_path']} --hyp {params['hyp_path']} --batch {params['batch']} --epochs {params['epochs']} --data {os.path.join(params['data_path'], 'data_local.yaml')} --weights {params['yolo_path']}/models/{version}_training.pt --name {modified_run_name}"
        subprocess.run(command, shell=True)

        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
        weights_path = f'{os.getcwd()}/runs/train/{modified_run_name}/weights/best.pt'
        destination_path = f'{os.getcwd()}/models/{version}_best_pothole.pt'
        os.system(f'cp {weights_path} {destination_path}')


def main():
    params = parse_yaml('src/param_YOLOv7.yaml')
    train_yolov7(params)


if __name__ == '__main__':
    main()
