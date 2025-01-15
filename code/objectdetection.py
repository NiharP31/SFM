import os
from pathlib import Path
import json
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, input_path, output_path, confidence=0.25):
        """
        Initialize the object detector
        Args:
            input_path: Path to preprocessed images
            output_path: Path to save detection results
            confidence: Confidence threshold for detections
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.confidence = confidence
        self.detection_output = self.output_path / "detections"
        self.visualization_output = self.output_path / "visualizations"
        
        # Create output directories
        self.detection_output.mkdir(parents=True, exist_ok=True)
        self.visualization_output.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using nano model for speed
        
        # KITTI relevant classes from COCO (YOLOv8 default classes)
        self.relevant_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            0: 'person',
            1: 'bicycle'
        }
    
    def detect_objects(self, image_path):
        """
        Detect objects in a single image
        Args:
            image_path: Path to input image
        Returns:
            detections: List of detection dictionaries
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return []
        
        # Run detection
        results = self.model(image, conf=self.confidence)[0]
        
        # Process detections
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            
            # Skip if not in relevant classes
            if cls_id not in self.relevant_classes:
                continue
            
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].item()
            
            detection = {
                'class': self.relevant_classes[cls_id],
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            }
            detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image_path, detections, output_path):
        """
        Visualize detections on image
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            output_path: Path to save visualization
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return
        
        # Colors for different classes
        colors = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0), # Blue
            'bus': (0, 0, 255),       # Red
            'truck': (255, 255, 0),   # Cyan
            'person': (255, 0, 255),  # Magenta
            'bicycle': (0, 255, 255)  # Yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls = det['class']
            conf = det['confidence']
            
            # Draw box
            color = colors.get(cls, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls} {conf:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        cv2.imwrite(str(output_path), image)
    
    def process_sequence(self):
        """Process all images in the sequence"""
        # Process each camera folder that exists
        for cam_idx in range(4):  # 4 cameras in KITTI
            cam_folder = self.input_path / f"image_{cam_idx:02d}"
            if not cam_folder.exists():
                continue
                
            print(f"\nProcessing camera {cam_idx}")
            
            # Create output folders for this camera
            cam_detections = self.detection_output / f"image_{cam_idx:02d}"
            cam_visualizations = self.visualization_output / f"image_{cam_idx:02d}"
            cam_detections.mkdir(parents=True, exist_ok=True)
            cam_visualizations.mkdir(parents=True, exist_ok=True)
            
            # Get all images
            image_files = sorted(list(cam_folder.glob("*.png")))
            
            # Process each image
            all_detections = {}
            for img_file in tqdm(image_files, desc=f"Camera {cam_idx}"):
                # Detect objects
                detections = self.detect_objects(img_file)
                
                # Save detections
                detection_file = cam_detections / f"{img_file.stem}.json"
                with open(detection_file, 'w') as f:
                    json.dump(detections, f, indent=2)
                
                # Save visualization
                vis_file = cam_visualizations / f"{img_file.stem}.png"
                self.visualize_detections(img_file, detections, vis_file)
                
                # Store detections for summary
                all_detections[img_file.stem] = detections
            
            # Save summary for this camera
            summary_file = self.detection_output / f"camera_{cam_idx}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_detections, f, indent=2)

def main():
    # Define paths
    input_path = "preprocessed_data"
    output_path = "detection_output"
    
    # Initialize detector
    detector = ObjectDetector(
        input_path=input_path,
        output_path=output_path,
        confidence=0.25  # Adjust confidence threshold as needed
    )
    
    try:
        # Run detection pipeline
        print("Starting object detection...")
        detector.process_sequence()
        
        print("\nObject detection complete!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    main()