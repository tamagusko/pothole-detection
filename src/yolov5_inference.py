import os
import subprocess
import yaml

def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params

def detect_yolov5(params):
    for version in params['versions']:
        command = f"python {params['yolo_path']}/detect.py --weights {os.getcwd()}/models/yolov5{version}_best.pt --img {params['img_size']} --source {params['data_path']} --project {params['project_name']} --name {params['run_name']}_{version} --save-txt"
        subprocess.run(command, shell=True)

def main():
    params = parse_yaml('param_detect_YOLOv5.yaml')
    detect_yolov5(params)

if __name__ == '__main__':
    main()

