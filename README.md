# Fire-and-Object-Detection-with-Distance

## Overview

This repository contains scripts and configuration files for a system that detects objects and identifies fire using YOLOv3 and RealSense depth data. The following files are included:

### YOLOv3 Files:
- **yolo_object_detection.py**: A Python script for object detection using YOLOv3. It uses the following supporting files:
  - `yolov3.cfg`: The configuration file for the YOLOv3 neural network.
  - `yolov3.weights`: The pre-trained weights for YOLOv3. (Size: 236.52 MB)
  - `coco.names`: A file containing the names of the COCO dataset classes used for object classification.

### Fire Detection Files:
- **fire1.py**: A script for detecting fire from images or video input.
- **updated_fire1.py**: An updated version of the fire detection script with enhanced functionality.

### Depth and Distance Calculation:
- **realsense_depth.py**: A script that interfaces with the Intel RealSense camera to get depth data.
- **detect distance.py**: A script that calculates the distance between the camera and detected objects.

## Setup

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv3 weights if not already available:
    - You can download the YOLOv3 pre-trained weights from [here](https://pjreddie.com/media/files/yolov3.weights) and place them in the repository directory.

## Usage

### Object Detection
To perform object detection using YOLOv3, run the following command:
```bash
python yolo_object_detection.py
```

### Fire Detection
To run the fire detection system, use:
```bash
python fire1.py
```
or the updated version:
```bash
python updated_fire1.py
```

### Depth and Distance Calculation
To run the depth detection system with Intel RealSense:
```bash
python realsense_depth.py
```
For distance calculation, run:
```bash
python detect\ distance.py
```

## Notes

- Ensure that you have an Intel RealSense camera for the depth measurement script.
- The fire detection system is based on analyzing visual cues in the video or image feed.

## COCO Dataset Classes

The file `coco.names` contains the following classes:
```
person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
... (truncated for brevity)
```

## License
This project is open-source and available under the MIT License.
