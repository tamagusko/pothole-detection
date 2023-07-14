import os
import subprocess
import yaml

def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params

def train_yolov8(params):
    for version in params['versions']:
        modified_run_name = f'{params["run_name"]}_YOLOv8{version}'
        command = f"python {os.path.join(params['yolo_path'], 'train.py')} --img {params['img_size']} --batch {params['batch']} --epochs {params['epochs']} --data {os.path.join(params['data_path'], 'data_local.yaml')} --weights yolov8{version}.pt --project {params['project_name']} --name {modified_run_name}"
        subprocess.run(command, shell=True)

        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
        os.system(f'mv {os.getcwd()}/yolov8{version}.pt {os.getcwd()}/models/yolov8{version}.pt')
        os.system(f'cp {params["project_name"]}/{modified_run_name}/weights/best.pt {os.getcwd()}/models/yolov8{version}_best.pt')

def main():
    params = parse_yaml('param_train_YOLOv8.yaml')
    train_yolov8(params)

if __name__ == '__main__':
    main()

