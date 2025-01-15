import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

class PipelineEvaluator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.detection_path = self.base_path / "detection_output" / "detections"
        self.segmentation_path = self.base_path / "segmentation_output" 
        self.report_path = self.base_path / "evaluation_report"
        
        # Create report directory
        self.report_path.mkdir(parents=True, exist_ok=True)

    def evaluate_detections(self):
        """Evaluate object detection results"""
        print("Evaluating object detections...")
        detection_dir = self.detection_path / "image_02"
        
        detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_distribution': {},
            'confidence_stats': {
                'mean': 0,
                'std': 0,
                'by_class': {}
            },
            'detections_per_frame': []
        }
        
        if not detection_dir.exists():
            print(f"Warning: Detection directory not found at {detection_dir}")
            return detection_stats
        
        # Get all JSON files
        detection_files = list(detection_dir.glob("*.json"))
        confidences = []
        
        for det_file in tqdm(detection_files, desc="Processing detections"):
            try:
                with open(det_file, 'r') as f:
                    detections = json.load(f)
                
                detection_stats['total_frames'] += 1
                detection_stats['total_detections'] += len(detections)
                detection_stats['detections_per_frame'].append(len(detections))
                
                # Process each detection
                for det in detections:
                    cls = det['class']
                    conf = det['confidence']
                    confidences.append(conf)
                    
                    # Update class distribution
                    detection_stats['class_distribution'][cls] = \
                        detection_stats['class_distribution'].get(cls, 0) + 1
                    
                    # Update class-wise confidence stats
                    if cls not in detection_stats['confidence_stats']['by_class']:
                        detection_stats['confidence_stats']['by_class'][cls] = []
                    detection_stats['confidence_stats']['by_class'][cls].append(conf)
            except Exception as e:
                print(f"Error processing {det_file}: {e}")
                continue
        
        # Calculate confidence statistics
        if confidences:
            detection_stats['confidence_stats']['mean'] = np.mean(confidences)
            detection_stats['confidence_stats']['std'] = np.std(confidences)
            
            # Calculate class-wise confidence stats
            for cls in detection_stats['confidence_stats']['by_class']:
                class_confs = detection_stats['confidence_stats']['by_class'][cls]
                detection_stats['confidence_stats']['by_class'][cls] = {
                    'mean': np.mean(class_confs),
                    'std': np.std(class_confs)
                }
        
        return detection_stats
    
    def evaluate_segmentation(self):
        """Evaluate segmentation results"""
        print("Evaluating segmentation results...")
        mask_dir = self.segmentation_path / "masks" / "image_02"
        
        segmentation_stats = {
            'total_frames': 0,
            'class_distribution': {},
            'pixel_coverage': {},
            'class_consistency': []
        }
        
        if not mask_dir.exists():
            print(f"Warning: Segmentation directory not found at {mask_dir}")
            return segmentation_stats
        
        # Process each frame's segmentation
        mask_files = list(mask_dir.glob("*_mask.npz"))
        
        for mask_file in tqdm(mask_files, desc="Processing segmentation"):
            try:
                mask_data = np.load(mask_file)
                mask = mask_data['mask']
                
                segmentation_stats['total_frames'] += 1
                
                # Calculate class distribution and pixel coverage
                unique, counts = np.unique(mask, return_counts=True)
                total_pixels = mask.size
                
                for cls, count in zip(unique, counts):
                    cls_str = str(int(cls))
                    
                    # Update class distribution
                    segmentation_stats['class_distribution'][cls_str] = \
                        segmentation_stats['class_distribution'].get(cls_str, 0) + 1
                    
                    # Update pixel coverage
                    if cls_str not in segmentation_stats['pixel_coverage']:
                        segmentation_stats['pixel_coverage'][cls_str] = []
                    segmentation_stats['pixel_coverage'][cls_str].append(count / total_pixels)
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
                continue
        
        # Calculate average pixel coverage
        for cls in segmentation_stats['pixel_coverage']:
            coverage = segmentation_stats['pixel_coverage'][cls]
            segmentation_stats['pixel_coverage'][cls] = {
                'mean': np.mean(coverage),
                'std': np.std(coverage)
            }
        
        return segmentation_stats
    
    def create_visualization_plots(self, detection_stats, segmentation_stats):
        """Create visualization plots for the report"""
        print("Creating visualization plots...")
        
        # Plot detection class distribution
        if detection_stats['class_distribution']:
            plt.figure(figsize=(12, 6))
            classes = list(detection_stats['class_distribution'].keys())
            counts = list(detection_stats['class_distribution'].values())
            plt.bar(classes, counts)
            plt.title('Detection Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.report_path / 'detection_class_distribution.png')
            plt.close()
            
            # Plot confidence distribution
            plt.figure(figsize=(12, 6))
            for cls, stats in detection_stats['confidence_stats']['by_class'].items():
                values = stats if isinstance(stats, list) else [stats['mean']]
                plt.hist(values, label=cls, alpha=0.5, bins=20)
            plt.title('Confidence Distribution by Class')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.report_path / 'confidence_distribution.png')
            plt.close()
        
        # Plot segmentation class coverage
        if segmentation_stats['pixel_coverage']:
            plt.figure(figsize=(12, 6))
            classes = list(segmentation_stats['pixel_coverage'].keys())
            means = [stats['mean'] for stats in segmentation_stats['pixel_coverage'].values()]
            stds = [stats['std'] for stats in segmentation_stats['pixel_coverage'].values()]
            plt.bar(classes, means, yerr=stds)
            plt.title('Average Class Pixel Coverage')
            plt.xlabel('Class')
            plt.ylabel('Coverage Ratio')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.report_path / 'segmentation_coverage.png')
            plt.close()
    
    def generate_report(self, detection_stats, segmentation_stats):
        """Generate evaluation report"""
        print("Generating report...")
        
        class_names = {
            '0': 'background',
            '1': 'person',
            '2': 'bicycle',
            '3': 'car',
            '4': 'motorcycle',
            '6': 'bus',
            '7': 'train',
            '8': 'truck',
            '13': 'bench',
            '14': 'bird',
            '15': 'cat',
            '16': 'dog',
            '17': 'horse',
            '18': 'sheep',
            '19': 'cow'
        }
        
        report = f"""
# Pipeline Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Object Detection Evaluation

### 1.1 General Statistics
- Total Frames Processed: {detection_stats['total_frames']}
- Total Detections: {detection_stats['total_detections']}
- Average Detections per Frame: {np.mean(detection_stats['detections_per_frame']) if detection_stats['detections_per_frame'] else 0:.2f}
- Overall Confidence: {detection_stats['confidence_stats']['mean']:.3f} ± {detection_stats['confidence_stats']['std']:.3f}

### 1.2 Class Distribution
"""
        
        if detection_stats['class_distribution']:
            for cls, count in detection_stats['class_distribution'].items():
                conf_stats = detection_stats['confidence_stats']['by_class'][cls]
                report += f"- {cls}: {count} detections\n"
                report += f"  - Average Confidence: {conf_stats['mean']:.3f} ± {conf_stats['std']:.3f}\n"
        
        report += """
## 2. Segmentation Evaluation

### 2.1 General Statistics
"""
        
        report += f"- Total Frames Processed: {segmentation_stats['total_frames']}\n"
        report += "\n### 2.2 Class Coverage\n"
        
        if segmentation_stats['pixel_coverage']:
            for cls, coverage in segmentation_stats['pixel_coverage'].items():
                class_name = class_names.get(cls, f'Class {cls}')
                report += f"- {class_name}:\n"
                report += f"  - Average Coverage: {coverage['mean']:.3f} ± {coverage['std']:.3f}\n"
        
        report += """
## 3. Challenges and Recommendations

### 3.1 Detection Challenges
- Classes with low detection counts might need additional training data
- Consider adjusting confidence threshold for optimal performance

### 3.2 Segmentation Challenges
- Class imbalance in pixel coverage might affect model performance
- Background dominates the segmentation, which is expected for road scenes

### 3.3 Recommendations
1. Fine-tune detection model for better performance on rare classes
2. Consider using weighted loss for segmentation to handle class imbalance
3. Implement temporal consistency checks
4. Validate results against ground truth if available

## 4. Visualization

Plots have been saved in the report directory:
1. detection_class_distribution.png
2. confidence_distribution.png
3. segmentation_coverage.png
"""
        
        # Save report
        with open(self.report_path / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {self.report_path / 'evaluation_report.md'}")

def main():
    # Define base path containing all stage outputs
    base_path = "C:/Users/nihar/Documents/github/3D_nerf_sfm"  # Current directory
    
    # Initialize evaluator
    evaluator = PipelineEvaluator(base_path)
    
    try:
        # Run evaluation
        detection_stats = evaluator.evaluate_detections()
        segmentation_stats = evaluator.evaluate_segmentation()
        
        # Create visualization plots
        evaluator.create_visualization_plots(detection_stats, segmentation_stats)
        
        # Generate report
        evaluator.generate_report(detection_stats, segmentation_stats)
        
        print("\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()