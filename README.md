# 3D Scene Understanding Pipeline for KITTI Dataset

A comprehensive pipeline for processing KITTI dataset images through preprocessing, 3D reconstruction, object detection, semantic segmentation, and evaluation.

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
Download the KITTI dataset from the official website[10](https://www.cvlibs.net/datasets/kitti/raw_data.php). The expected folder structure is:
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
This project is licensed under the MIT License.

## Acknowledgments
- KITTI Dataset[10]
- COLMAP Structure-from-Motion[11]
- YOLOv8 by Ultralytics[6]

Citations:
[1] https://pplx-res.cloudinary.com/image/upload/v1736901335/user_uploads/rvfPxBmjCkcbibZ/image.jpg
[2] https://pplx-res.cloudinary.com/image/upload/v1736900152/user_uploads/iXbjbGGkgnBpVpb/image.jpg
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/29390211/ac418235-52b7-41cd-becc-1365bc29d1a1/paste.txt
[4] https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html
[5] https://colmap.github.io/tutorial.html
[6] https://keylabs.ai/blog/mastering-object-detection-with-yolov8/
[7] https://mmdetection3d.readthedocs.io/en/v0.17.3/datasets/kitti_det.html
[8] https://colmap.github.io/faq.html
[9] https://www.geeksforgeeks.org/object-detection-using-yolov8/
[10] https://github.com/Armanasq/kitti-dataset-tutorial
[11] https://demuc.de/colmap/
[12] https://www.digitalocean.com/community/tutorials/yolov8-a-revolutionary-advancement-in-object-detection-2
[13] http://semantic-kitti.org/dataset.html
[14] https://www.digitalocean.com/community/tutorials/photogrammetry-pipeline-on-gpu-droplet
[15] https://blog.roboflow.com/how-to-detect-objects-with-yolov8/
[16] https://www.cvlibs.net/datasets/kitti-360/documentation.php
[17] https://www.cs.cmu.edu/~reconstruction/colmap.html
[18] https://www.cvlibs.net/publications/Geiger2013IJRR.pdf
[19] https://colmap.github.io
[20] https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/ppj2.20068