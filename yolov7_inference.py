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


def detect_yolov7(params):
    """Detect objects with different versions of YOLOv7.

    Args:
        params (dict): Dictionary of parameters parsed from yaml file.
    """
    for version in params['versions']:
        # Run the detect.py script with the specified parameters
        command = f"python {params['yolo_path']}/detect.py --weights {os.getcwd()}/models/yolov7{version}_best.pt --img {params['img_size']} --source {params['data_path']} --project {params['project_name']} --name {params['run_name']}_{version} --save-txt"
        subprocess.run(command, shell=True)


def main():
    """Main function."""
    params = parse_yaml('param_train_YOLOv7.yaml')
    detect_yolov7(params)


if __name__ == '__main__':
    main()
