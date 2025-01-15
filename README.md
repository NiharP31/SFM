# 3D Scene Understanding Pipeline for KITTI Dataset

A comprehensive pipeline for processing KITTI dataset images through preprocessing, 3D reconstruction, object detection, semantic segmentation, and evaluation.

<video width="600" controls>
  <source src="vizualization_output/sequence_visualization.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Overview

This project implements a complete pipeline for analyzing KITTI autonomous driving data, including:
- Image preprocessing and standardization
- 3D scene reconstruction using COLMAP
- Object detection using YOLOv8
- Semantic segmentation
- Pipeline evaluation and visualization

## Prerequisites

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- OpenCV
- NumPy
- COLMAP
- Ultralytics (YOLOv8)
- Open3D
- Matplotlib
- tqdm

### Dataset
Download the KITTI dataset from the official [website](https://www.cvlibs.net/datasets/kitti/raw_data.php). The expected folder structure is:
```
data/
├── 2011_09_26_calib/
├── 2011_09_26_drive_0009_sync/
│   ├── image_00/
│   ├── image_01/
│   ├── image_02/
│   └── image_03/
└── timestamps.txt
```

## Pipeline Components

### 1. Data Preprocessing
- Standardizes image sizes
- Organizes data structure
- Preserves timestamp information

### 2. 3D Reconstruction
- Feature extraction and matching
- Sparse reconstruction
- Dense reconstruction
- Camera pose estimation

### 3. Object Detection
- YOLOv8-based detection
- Multiple object class support
- Confidence-based filtering

### 4. Semantic Segmentation
- Per-pixel semantic labeling
- Multi-class segmentation
- Instance segmentation support

### 5. Evaluation
- Detection accuracy metrics
- Segmentation quality assessment
- Visualization generation
- Comprehensive reporting

## Usage

1. Preprocess the data:
```python
python preprocess.py --data_root data --sequence 2011_09_26_drive_0009_sync
```

2. Run 3D reconstruction:
```python
python reconstruct.py --input preprocessed_data --output reconstruction_output
```

3. Perform object detection:
```python
python detect.py --input preprocessed_data --output detection_output
```

4. Run semantic segmentation:
```python
python segment.py --input preprocessed_data --output segmentation_output
```

5. Generate evaluation report:
```python
python evaluate.py --base_path .
```

## Output Structure
```
./
├── preprocessed_data/
├── reconstruction_output/
├── detection_output/
├── segmentation_output/
└── evaluation_report/
```

## License
This project is licensed under the [MIT](https://github.com/NiharP31/SFM/blob/main/LICENSE) License.

## Acknowledgments
- KITTI Dataset [Link](https://www.cvlibs.net/datasets/kitti/user_login.php)
- COLMAP Structure-from-Motion [Link](https://demuc.de/colmap/)
- YOLOv8 by Ultralytics [Link](https://docs.ultralytics.com/models/yolov8/)