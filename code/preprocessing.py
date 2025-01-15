import os
import cv2
import numpy as np
from tqdm import tqdm

class KITTIPreprocessor:
    def __init__(self, data_root, sequence_folder, output_path, target_size=(832, 256)):
        """
        Initialize the KITTI preprocessor
        Args:
            data_root: Root path to the KITTI dataset
            sequence_folder: Name of the sequence folder
            output_path: Path to save preprocessed data
            target_size: Target image size (width, height)
        """
        self.data_root = data_root
        self.sequence_folder = sequence_folder
        self.output_path = output_path
        self.target_size = target_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
    def preprocess_images(self):
        """Process all images in the sequence"""
        # Path to image folders
        image_folders = [
            os.path.join(self.data_root, self.sequence_folder, 'image_00', 'data'),
            os.path.join(self.data_root, self.sequence_folder, 'image_01', 'data'),
            os.path.join(self.data_root, self.sequence_folder, 'image_02', 'data'),
            os.path.join(self.data_root, self.sequence_folder, 'image_03', 'data')
        ]
        
        # Process each camera folder
        for i, folder in enumerate(image_folders):
            if not os.path.exists(folder):
                print(f"Skipping camera {i}, folder not found: {folder}")
                continue
                
            print(f"\nProcessing camera {i}")
            
            # Create output subfolder for this camera
            cam_output = os.path.join(self.output_path, f'image_{i:02d}')
            os.makedirs(cam_output, exist_ok=True)
            
            # Get all PNG files in this folder
            image_files = [f for f in os.listdir(folder) if f.endswith('.png')]
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"Camera {i}"):
                # Read image
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize image
                    resized_img = cv2.resize(img, self.target_size)
                    
                    # Save processed image
                    output_path = os.path.join(cam_output, img_file)
                    cv2.imwrite(output_path, resized_img)

    def save_timestamps(self):
        """Copy timestamp files"""
        timestamp_files = [
            os.path.join(self.data_root, self.sequence_folder, 'image_00', 'timestamps.txt'),
            os.path.join(self.data_root, self.sequence_folder, 'image_01', 'timestamps.txt'),
            os.path.join(self.data_root, self.sequence_folder, 'image_02', 'timestamps.txt'),
            os.path.join(self.data_root, self.sequence_folder, 'image_03', 'timestamps.txt')
        ]

        # Create timestamps directory in output
        timestamps_dir = os.path.join(self.output_path, 'timestamps')
        os.makedirs(timestamps_dir, exist_ok=True)

        # Copy each timestamp file
        for i, ts_file in enumerate(timestamp_files):
            if os.path.exists(ts_file):
                output_file = os.path.join(timestamps_dir, f'image_{i:02d}_timestamps.txt')
                with open(ts_file, 'r') as src, open(output_file, 'w') as dst:
                    dst.write(src.read())
                print(f"Copied timestamps for camera {i}")

def main():
    # Define paths based on your folder structure
    data_root = "data"  # Root folder containing all KITTI data
    sequence_folder = "2011_09_26_drive_0009_sync"  # Specific sequence folder
    output_path = "preprocessed_data"  # Output folder for processed data
    
    # Initialize preprocessor
    preprocessor = KITTIPreprocessor(
        data_root=data_root,
        sequence_folder=sequence_folder,
        output_path=output_path
    )
    
    # Process images
    print("Starting image preprocessing...")
    preprocessor.preprocess_images()
    
    # Save timestamps
    print("\nSaving timestamps...")
    preprocessor.save_timestamps()
    
    print("\nPreprocessing complete!")
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    main()