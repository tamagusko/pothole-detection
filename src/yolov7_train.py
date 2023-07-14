import os
import subprocess
import yaml

def parse_yaml(yaml_file):
    with open(yaml_file) as file:
        params = yaml.safe_load(file)
    return params

def train_yolov7(params):
    for version in params['versions']:
        command = f"python {params['yolo_path']}/train.py --img-size {params['img_size']} --cfg {params['cfg_path']} --hyp {params['hyp_path']} --batch {params['batch']} --epochs {params['epochs']} --data {params['data_path']} --weights {params['yolo_path']}/models/{version}_training.pt --name {params['run_name']}_{version}"
        subprocess.run(command, shell=True)

        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
        os.system(f"mv {params['yolo_path']}/models/{version}_training.pt {os.getcwd()}/models/yolov7{version}.pt")
        os.system(f"cp {params['yolo_path']}/runs/train/{params['run_name']}_{version}/weights/best.pt {os.getcwd()}/models/yolov7{version}_best.pt")

def main():
    params = parse_yaml('param_train_YOLOv7.yaml')
    train_yolov7(params)

if __name__ == '__main__':
    main()

