# TLDR

This README provides instructions on how to train and infer different versions of YOLO on your dataset.

## YOLOv5

Adjust patches and names in path/to/pothole-detection/src/param_train_YOLOv5.yaml

```bash
cd path/to/pothole-detection/
python python src/yolov5_train.py # train
python python src/yolov5_inference.py # infer models
```

## YOLOv7

Adjust patches and names in path/to/pothole-detection/src/param_train_YOLOv7.yaml

```bash
cd path/to/pothole-detection/
python python src/yolov7_train.py # train
python python src/yolov7_inference.py # infer models
```

## YOLOv8

Adjust patches and names in path/to/pothole-detection/src/param_train_YOLOv8.yaml

```bash
cd path/to/pothole-detection/
python python src/yolov8_train.py # train
python python src/yolov8_inference.py # infer models
```

## Support

For issues, bug reports, and pull requests, please use this GitHub page. To contact me directly, send an email to <tamagusko@gmail.com>.
