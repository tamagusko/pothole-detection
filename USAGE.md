# Training YOLO Models

This README provides instructions on how to train different versions of YOLO on your dataset.

## YOLOv5

The following are instructions for training YOLOv5 with different configurations:

YOLOv5:

```bash
python path/to/yolov5/train.py --img 300 --batch 16 --epochs 300 --data path/to/data/data_local.yaml --weights yolov5VERSION.pt --project 'project_name' --name 'run_name'
```

NOTE: Change the VERSION to the specific version of YOLOv5 (n, s, m, l, x).

YOLOv7:

```bash
python path/to/yolov5/train.py --img-size 300 --batch 16 --epochs 300 --cfg  path/to/cfg/training/yolov7.yaml --hyp data/hyp.scratch.custom.yaml --data path/to/data/data_local.yaml --weights yolov7_training.pt --project 'project_name' --name 'run_name'
```

NOTE: Change the VERSION to the specific version of YOLOv7 (yolov7_training.pt) or YOLOv7x (yolov7x_training.pt)

YOLOv5:

```bash
python path/to/yolov5/train.py --img 300 --batch 16 --epochs 300 --data path/to/data/data_local.yaml --weights yolov5VERSION.pt --project 'project_name' --name 'run_name'
```

NOTE: Change the VERSION to the specific version of YOLOv8 (n, s, m, l, x).
