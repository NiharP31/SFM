import os
import subprocess
import numpy as np
import json
from pathlib import Path
import shutil

class SceneReconstructor:
    def __init__(self, input_path, output_path):
        """
        Initialize the 3D scene reconstructor
        Args:
            input_path: Path to preprocessed images
            output_path: Path to save reconstruction results
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.colmap_output = self.output_path / "colmap"
        self.sparse_dir = self.colmap_output / "sparse"
        self.dense_dir = self.colmap_output / "dense"
        
        # Create output directories
        self.colmap_output.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
    def run_colmap_feature_extractor(self):
        """Run COLMAP feature extraction"""
        print("Extracting features...")
        
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.colmap_output / "database.db"),
            "--image_path", str(self.input_path / "image_02"),  # Using left color camera
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1"
        ]
        
        subprocess.run(cmd, check=True)
        
    def run_colmap_matcher(self):
        """Run COLMAP feature matching"""
        print("Matching features...")
        
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.colmap_output / "database.db"),
            "--SiftMatching.use_gpu", "1"
        ]
        
        subprocess.run(cmd, check=True)
        
    def run_colmap_sparse_reconstruction(self):
        """Run COLMAP sparse reconstruction"""
        print("Running sparse reconstruction...")
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.colmap_output / "database.db"),
            "--image_path", str(self.input_path / "image_02"),
            "--output_path", str(self.sparse_dir),
            "--Mapper.min_num_matches", "10",
            "--Mapper.init_min_tri_angle", "4.0",
            "--Mapper.multiple_models", "1",
        ]
        
        subprocess.run(cmd, check=True)

    def prepare_dense_workspace(self):
        """Prepare workspace for dense reconstruction"""
        print("Preparing dense reconstruction workspace...")

        # Create necessary directories
        workspace_dir = self.dense_dir / "workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)  # Clean existing workspace
        workspace_dir.mkdir(parents=True)
        
        sparse_dir = workspace_dir / "sparse"
        sparse_dir.mkdir(parents=True)
        
        images_dir = workspace_dir / "images"
        images_dir.mkdir(parents=True)
        
        stereo_dir = workspace_dir / "stereo"
        stereo_dir.mkdir(parents=True)

        # Find the first reconstruction (usually 0)
        sparse_recon_dir = next(self.sparse_dir.glob("[0-9]"))
        if not sparse_recon_dir.exists():
            raise RuntimeError("No sparse reconstruction found")

        # Convert binary to text format
        cmd_convert = [
            "colmap", "model_converter",
            "--input_path", str(sparse_recon_dir),
            "--output_path", str(sparse_dir),
            "--output_type", "TXT"
        ]
        subprocess.run(cmd_convert, check=True)

        # Copy images
        for img_file in (self.input_path / "image_02").glob("*.png"):
            shutil.copy2(img_file, images_dir)

        # Create stereo config
        with open(stereo_dir / "patch-match.cfg", "w") as f:
            f.write("__auto__, 5\n")  # Reduced number of image pairs for stability

        return workspace_dir

    def run_colmap_dense_reconstruction(self):
        """Run COLMAP dense reconstruction"""
        print("Running dense reconstruction...")
        
        try:
            # Prepare workspace
            workspace_dir = self.prepare_dense_workspace()
            
            # Run MVS pipeline in smaller steps
            
            # 1. Image undistortion with more conservative settings
            print("Undistorting images...")
            cmd_undistort = [
                "colmap", "image_undistorter",
                "--image_path", str(workspace_dir / "images"),
                "--input_path", str(workspace_dir / "sparse"),
                "--output_path", str(workspace_dir),
                "--output_type", "COLMAP",
                "--max_image_size", "2048"  # Limit image size
            ]
            subprocess.run(cmd_undistort, check=True)
            
            # 2. Patch match stereo with conservative memory settings
            print("Running patch match stereo...")
            cmd_stereo = [
                "colmap", "patch_match_stereo",
                "--workspace_path", str(workspace_dir),
                "--PatchMatchStereo.max_image_size", "1024",  # Further reduced for stability
                "--PatchMatchStereo.window_radius", "5",
                "--PatchMatchStereo.window_step", "2",
                "--PatchMatchStereo.num_samples", "7",
                "--PatchMatchStereo.num_iterations", "3",
                "--PatchMatchStereo.geom_consistency", "true",
                "--PatchMatchStereo.filter", "true",
                "--PatchMatchStereo.filter_min_ncc", "0.1",
                "--PatchMatchStereo.filter_min_triangulation_angle", "3.0",
                "--PatchMatchStereo.cache_size", "8"  # Reduced memory usage
            ]
            subprocess.run(cmd_stereo, check=True)
            
            # 3. Stereo fusion with conservative settings
            print("Running stereo fusion...")
            cmd_fusion = [
                "colmap", "stereo_fusion",
                "--workspace_path", str(workspace_dir),
                "--output_path", str(self.dense_dir / "fused.ply"),
                "--input_type", "geometric",
                "--StereoFusion.min_num_pixels", "5",
                "--StereoFusion.max_reproj_error", "2.0",
                "--StereoFusion.max_depth_error", "0.1",
                "--StereoFusion.max_normal_error", "20"
            ]
            subprocess.run(cmd_fusion, check=True)
            
            print("Dense reconstruction completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError in dense reconstruction step: {e}")
            print("Detailed error info:")
            print(f"Command that failed: {e.cmd}")
            print(f"Return code: {e.returncode}")
            if e.output:
                print(f"Output: {e.output.decode()}")
            raise
        except Exception as e:
            print(f"\nUnexpected error in dense reconstruction: {e}")
            raise
        
    def save_camera_poses(self):
        """Extract and save camera poses from COLMAP reconstruction"""
        print("Saving camera poses...")
        
        workspace_dir = self.dense_dir / "workspace" / "sparse"
        
        # Read cameras and images from COLMAP text files
        cameras = {}
        camera_file = workspace_dir / "cameras.txt"
        if camera_file.exists():
            with open(camera_file, "r") as f:
                next(f)  # Skip count line
                for line in f:
                    if line[0] != "#":
                        camera_id, model, width, height, *params = line.split()
                        cameras[camera_id] = {
                            "model": model,
                            "width": int(width),
                            "height": int(height),
                            "params": [float(p) for p in params]
                        }
        
        images = {}
        image_file = workspace_dir / "images.txt"
        if image_file.exists():
            with open(image_file, "r") as f:
                next(f)  # Skip count line
                for line in f:
                    if line[0] != "#":
                        data = line.split()
                        image_id = data[0]
                        qw, qx, qy, qz = map(float, data[1:5])
                        tx, ty, tz = map(float, data[5:8])
                        camera_id = data[8]
                        image_name = data[9]
                        
                        images[image_id] = {
                            "name": image_name,
                            "camera_id": camera_id,
                            "rotation": [qw, qx, qy, qz],
                            "translation": [tx, ty, tz]
                        }
        
        # Save as JSON
        camera_data = {
            "cameras": cameras,
            "images": images
        }
        
        with open(self.colmap_output / "camera_poses.json", "w") as f:
            json.dump(camera_data, f, indent=2)

def main():
    # Define paths
    input_path = "preprocessed_data"
    output_path = "reconstruction_output"
    
    # Initialize reconstructor
    reconstructor = SceneReconstructor(input_path, output_path)
    
    try:
        # Run reconstruction pipeline
        reconstructor.run_colmap_feature_extractor()
        reconstructor.run_colmap_matcher()
        reconstructor.run_colmap_sparse_reconstruction()
        reconstructor.run_colmap_dense_reconstruction()
        reconstructor.save_camera_poses()
        
        print("\nReconstruction complete!")
        print(f"Results saved to: {output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during reconstruction: {e}")
        print("Make sure COLMAP is installed and accessible from command line")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()