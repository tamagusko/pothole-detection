"""YOLOv5 training script for multiple versions."""

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


def train_yolov5(yolo_path, img_size, batch, epochs, data_path, project_name, run_name, versions):
    """Train different versions of YOLOv5.

    Args:
        yolo_path (str): Path to the YOLOv5 scripts.
        img_size (int): Size of the image for training.
        batch (int): Batch size for training.
        epochs (int): Number of epochs for training.
        data_path (str): Path to the training data.
        project_name (str): Name of the project.
        run_name (str): Name of the run.
        versions (list[str]): List of YOLOv5 versions to be trained.
    """
    for version in versions:
        modified_run_name = f'{run_name}_YOLOv5{version}'
        command = f"python {os.path.join(yolo_path, 'train.py')} --img {img_size} --batch {batch} --epochs {epochs} --data {os.path.join(data_path, 'data_local.yaml')} --weights yolov5{version}.pt --project {project_name} --name {modified_run_name}"
        subprocess.run(command, shell=True)

        # Creating models subfolder in current directory if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)

        # Moving the base model to models folder in current directory
        os.system(
            f'mv {os.getcwd()}/yolov5{version}.pt {os.getcwd()}/models/yolov5{version}.pt',
        )

        # Copying the best model to models folder in current directory with specific naming convention
        os.system(
            f'cp {project_name}/{modified_run_name}/weights/best.pt {os.getcwd()}/models/yolov5{version}_best.pt',
        )


def main():
    """Main function."""
    params = parse_yaml('param_train_YOLOv5.yaml')
    yolo_path = params['yolo_path']
    img_size = params['img_size']
    batch = params['batch']
    epochs = params['epochs']
    data_path = params['data_path']
    project_name = params['project_name']
    run_name = params['run_name']
    versions = params['versions']

    train_yolov5(
        yolo_path, img_size, batch, epochs,
        data_path, project_name, run_name, versions,
    )


if __name__ == '__main__':
    main()