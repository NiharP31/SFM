import os
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import cv2

class SemanticSegmenter:
    def __init__(self, input_path, output_path):
        """
        Initialize the semantic segmenter
        Args:
            input_path: Path to preprocessed images
            output_path: Path to save segmentation results
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.mask_output = self.output_path / "masks"
        self.visualization_output = self.output_path / "visualizations"
        
        # Create output directories
        self.mask_output.mkdir(parents=True, exist_ok=True)
        self.visualization_output.mkdir(parents=True, exist_ok=True)
        
        # Initialize DeepLabV3 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # COCO classes relevant to KITTI
        self.relevant_classes = {
            0: 'background',
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            8: 'truck',
            7: 'train',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow'
        }
        
        # Color map for visualization
        self.color_map = {
            0: [0, 0, 0],        # background: black
            1: [220, 20, 60],    # person: red
            2: [119, 11, 32],    # bicycle: dark red
            3: [0, 0, 142],      # car: blue
            4: [0, 0, 230],      # motorcycle: bright blue
            6: [0, 60, 100],     # bus: dark blue
            8: [0, 0, 70],       # truck: darker blue
            7: [0, 80, 100],     # train: blue variant
            13: [255, 255, 255], # bench: white
            14: [190, 153, 153], # bird: light pink
            15: [250, 170, 30],  # cat: orange
            16: [220, 220, 0],   # dog: yellow
            17: [152, 251, 152], # horse: pale green
            18: [70, 130, 180],  # sheep: steel blue
            19: [220, 20, 60],   # cow: red
        }

    def segment_image(self, image_path):
        """
        Perform semantic segmentation on a single image
        Args:
            image_path: Path to input image
        Returns:
            mask: Segmentation mask
            colored_mask: Visualization of the mask
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        mask = torch.argmax(output, dim=0).cpu().numpy()
        
        # Create colored mask for visualization
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_idx, color in self.color_map.items():
            colored_mask[mask == class_idx] = color
            
        return mask, colored_mask
    
    def create_overlay(self, image_path, colored_mask, alpha=0.5):
        """
        Create an overlay of the original image and segmentation mask
        """
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (colored_mask.shape[1], colored_mask.shape[0]))
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        return overlay

    def save_segmentation_data(self, mask, image_name, output_dir):
        """
        Save segmentation data in a compressed format
        """
        # Save mask as compressed numpy array
        mask_file = output_dir / f"{image_name}_mask.npz"
        np.savez_compressed(mask_file, mask=mask)
        
        # Create and save metadata
        unique_classes = np.unique(mask)
        metadata = {
            'present_classes': [
                {
                    'class_id': int(class_id),
                    'class_name': self.relevant_classes.get(int(class_id), 'unknown'),
                    'pixel_count': int(np.sum(mask == class_id))
                }
                for class_id in unique_classes
                if class_id in self.relevant_classes
            ]
        }
        
        metadata_file = output_dir / f"{image_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_sequence(self):
        """Process all images in the sequence"""
        # Process each camera folder that exists
        for cam_idx in range(4):  # 4 cameras in KITTI
            cam_folder = self.input_path / f"image_{cam_idx:02d}"
            if not cam_folder.exists():
                continue
                
            print(f"\nProcessing camera {cam_idx}")
            
            # Create output folders for this camera
            cam_masks = self.mask_output / f"image_{cam_idx:02d}"
            cam_visualizations = self.visualization_output / f"image_{cam_idx:02d}"
            cam_masks.mkdir(parents=True, exist_ok=True)
            cam_visualizations.mkdir(parents=True, exist_ok=True)
            
            # Get all images
            image_files = sorted(list(cam_folder.glob("*.png")))
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"Camera {cam_idx}"):
                try:
                    # Perform segmentation
                    mask, colored_mask = self.segment_image(img_file)
                    
                    # Create and save overlay visualization
                    overlay = self.create_overlay(img_file, colored_mask)
                    vis_path = cam_visualizations / f"{img_file.stem}_overlay.png"
                    cv2.imwrite(str(vis_path), overlay)
                    
                    # Save mask and metadata
                    self.save_segmentation_data(mask, img_file.stem, cam_masks)
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue

def main():
    # Define paths
    input_path = "preprocessed_data"
    output_path = "segmentation_output"
    
    # Initialize segmenter
    segmenter = SemanticSegmenter(
        input_path=input_path,
        output_path=output_path
    )
    
    try:
        # Run segmentation pipeline
        print("Starting semantic segmentation...")
        segmenter.process_sequence()
        
        print("\nSegmentation complete!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during segmentation: {e}")

if __name__ == "__main__":
    main()