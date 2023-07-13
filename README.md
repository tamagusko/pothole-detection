# PAPER: Optimizing Pothole Detection in Pavements: A Comparative Analysis of Deep Learning Models

This repository contains the code and models related to the research paper "Optimizing Pothole Detection in Pavements: A Comparative Analysis of Deep Learning Models" by Tiago Tamagusko and Adelino Ferreira. This paper was presented at the Second International Conference on Maintenance and Rehabilitation of Constructed Infrastructure Facilities in Honolulu, HI, USA, 16–19 August 2023.

## TLDR

The paper delves into the exploration of state-of-the-art computer vision techniques for detecting pavement potholes, comparing the performance of several deep learning models based on the You Only Look Once (YOLO) family. The models were trained and tested on a dataset containing 665 road pavement images with labeled potholes. The findings revealed that YOLOv4 yielded the highest mean average precision (mAP), while YOLOv4-tiny offered the optimal reduced inference time, making it suitable for mobile applications. Additionally, the YOLOv5s model demonstrated potential by showcasing impressive results and ease of implementation and scalability.

## Installation

1. Clone the repository to your local machine.

    ```bash
    git clone https://github.com/tamagusko/pothole-detection.git
    ```

2. Create a Python virtual environment and activate it:

    ```bash
    python -m venv my_env_name
    source my_env_name/bin/activate
    ```

3. Install PyTorch:

    ```bash
    pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    ```

4. Clone YOLOv5:

    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```

5. Install the requirements:

    ```bash
    cd yolov5  
    pip install -r requirements.txt
    ```

## Usage

### Run Locally

To train the model:

```bash
python path/to/yolov5/train.py --img 300 --batch 16 --epochs 300 --data path/to/data/data_local.yaml --weights yolov5s.pt --project 'project_name' --name 'run_name'
```

### Run Using Docker

1. Pull the docker image:

    ```bash
    sudo docker pull ultralytics/yolov5:latest
    ```

2. Run the docker image:

    ```bash
    sudo docker run -v path/to/data/:/mnt/data --ipc=host -it --network host --gpus all ultralytics/yolov5:latest
    ```

3. Train the model:

    ```bash
    python train.py --img 300 --batch 16 --epochs 300 --data /mnt/data/data.yaml --weights yolov5s.pt --project "test" --name "run1"
    ```

## Citation

If you find this repository useful for your research, please cite our paper as follows:

```bibtex
@inproceedings{tamagusko2023optimizing,
  title={Optimizing Pothole Detection in Pavements: A Comparative Analysis of Deep Learning Models},
  author={Tamagusko, Tiago and Ferreira, Adelino},
  booktitle={Second International Conference on Maintenance and Rehabilitation of Constructed Infrastructure Facilities},
  year={2023},
  location={Honolulu, HI, USA}
}
```

## Support

For issues, bug reports, and pull requests, please use this GitHub page. To contact Tiago Tamagusko directly, send an email to <tamagusko@gmail.com>.

## License

This work is licensed under a [CC-BY-NC-ND-4.0](LICENSE) license. (c) 2023, [Tiago Tamagusko](https://tamagusko.com).
