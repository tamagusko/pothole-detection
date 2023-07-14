"""YOLOv7 training script for multiple versions."""

import os
import subprocess
import yaml


def parse_yaml(yaml_file):
    """Parse yaml file.

    Args:
        yaml_file (str): Path to the yaml file.

    Returns:
        dict: Parameters read from the yaml file.
    """
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params


def train_yolov7(params):
    """Train different versions of YOLOv7.

    Args:
        params (dict): Dictionary of parameters parsed from yaml file.
    """
    for version in params['versions']:
        command = f"python {params['yolo_path']}/train.py --img-size {params['img_size']} --cfg {params['cfg_path']} --hyp {params['hyp_path']} --batch {params['batch']} --epochs {params['epochs']} --data {params['data_path']} --weights {params['yolo_path']}/models/{version}_training.pt --name {params['run_name']}_{version}"
        subprocess.run(command, shell=True)


def main():
    """Main function."""
    params = parse_yaml('param_train_YOLOv7.yaml')
    train_yolov7(params)


if __name__ == "__main__":
    main()

