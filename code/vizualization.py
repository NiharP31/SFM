import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import cv2
from tqdm import tqdm

class PipelineIntegrator:
    def __init__(self, base_path):
        """
        Initialize the pipeline integrator
        Args:
            base_path: Base path containing all stage outputs
        """
        self.base_path = Path(base_path)
        self.preprocessed_path = self.base_path / "preprocessed_data"
        self.reconstruction_path = self.base_path / "reconstruction_output"
        self.detection_path = self.base_path / "detection_output"
        self.segmentation_path = self.base_path / "segmentation_output"
        self.visualization_path = self.base_path / "visualization_output"
        
        # Create visualization output directory
        self.visualization_path.mkdir(parents=True, exist_ok=True)
        
    def load_point_cloud(self):
        """Load the dense point cloud from COLMAP reconstruction"""
        print("Loading point cloud...")
        dense_path = self.reconstruction_path / "colmap/dense/fused.ply"
        if not dense_path.exists():
            raise FileNotFoundError(f"Point cloud not found at {dense_path}")
            
        pcd = o3d.io.read_point_cloud(str(dense_path))
        return pcd
    
    def load_camera_poses(self):
        """Load camera poses from COLMAP reconstruction"""
        print("Loading camera poses...")
        poses_path = self.reconstruction_path / "colmap/camera_poses.json"
        if not poses_path.exists():
            raise FileNotFoundError(f"Camera poses not found at {poses_path}")
            
        with open(poses_path, 'r') as f:
            camera_data = json.load(f)
        return camera_data
    
    def load_detections(self, frame_id):
        """Load object detections for a specific frame"""
        detection_path = self.detection_path / "image_02" / f"{frame_id}.json"
        if not detection_path.exists():
            return []
            
        with open(detection_path, 'r') as f:
            detections = json.load(f)
        return detections
    
    def load_segmentation(self, frame_id):
        """Load segmentation mask for a specific frame"""
        mask_path = self.segmentation_path / "masks/image_02" / f"{frame_id}_mask.npz"
        if not mask_path.exists():
            return None
            
        mask_data = np.load(mask_path)
        return mask_data['mask']

    def create_3d_visualization(self):
        """Create an integrated 3D visualization"""
        # Load point cloud
        pcd = self.load_point_cloud()
        
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # Optional: Add camera frustums from poses
        camera_data = self.load_camera_poses()
        for image_data in camera_data['images'].values():
            # Create camera frustum
            frustum = self.create_camera_frustum(image_data)
            vis.add_geometry(frustum)
        
        # Optimize view
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
        vis.get_view_control().set_zoom(0.8)
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    # def create_camera_frustum(self, image_data, scale=1.0):
    #     """Create a camera frustum mesh for visualization"""
    #     # Extract camera pose
    #     qw, qx, qy, qz = image_data['rotation']
    #     tx, ty, tz = image_data['translation']
        
    #     # Create frustum vertices
    #     points = np.array([
    #         [0, 0, 0],
    #         [-1, -1, 2],
    #         [1, -1, 2],
    #         [1, 1, 2],
    #         [-1, 1, 2]
    #     ]) * scale
        
    #     # Create frustum lines
    #     lines = np.array([
    #         [0, 1], [0, 2], [0, 3], [0, 4],
    #         [1, 2], [2, 3], [3, 4], [4, 1]
    #     ])
        
    #     # Create line set
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(points)
    #     line_set.lines = o3d.utility.Vector2iVector(lines)
        
    #     # Transform to camera pose
    #     # Convert quaternion and translation to transformation matrix
    #     # ... (add quaternion to matrix conversion)
        
    #     return line_set

    def create_camera_frustum(self, image_data, scale=0.5):
        """Create a camera frustum mesh for visualization"""
        # Extract camera pose
        qw, qx, qy, qz = image_data['rotation']
        tx, ty, tz = image_data['translation']
        
        # Create frustum vertices
        points = np.array([
            [0, 0, 0],         # Camera center
            [-1, -1, 2],       # Top-left
            [1, -1, 2],        # Top-right
            [1, 1, 2],         # Bottom-right
            [-1, 1, 2]         # Bottom-left
        ]) * scale
        
        # Create frustum lines
        lines = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Lines connecting corners
        ])
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Transform to camera pose
        # 1. Convert quaternion to rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_quaternion(
            [qw, qx, qy, qz]
        )
        
        # 2. Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R          # Set rotation part
        T[:3, 3] = [tx, ty, tz]  # Set translation part
        
        # 3. Apply transformation to line set
        line_set.transform(T)
        
        return line_set

    def create_frame_visualization(self, frame_id):
        """Create visualization for a single frame with all annotations"""
        # Load original image
        image_path = self.preprocessed_path / "image_02" / f"{frame_id}.png"
        if not image_path.exists():
            return None
            
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Load and overlay detections
        detections = self.load_detections(frame_id)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{det['class']} {det['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # # Load and overlay segmentation
        # mask = self.load_segmentation(frame_id)
        # if mask is not None:
        #     # Create colored mask
        #     colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        #     for class_id in np.unique(mask):
        #         color = np.random.randint(0, 255, 3).tolist()
        #         colored_mask[mask == class_id] = color
            
        #     # Overlay with transparency
        #     overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        # else:
        #     overlay = image
        
        # return overlay
        # Load and overlay segmentation
        mask = self.load_segmentation(frame_id)
        if mask is not None:
            # Use consistent color map
            color_map = {
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
            
            # Create colored mask using consistent color map
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_id in np.unique(mask):
                if class_id in color_map:
                    colored_mask[mask == class_id] = color_map[class_id]
                else:
                    colored_mask[mask == class_id] = [128, 128, 128]  # gray for unknown classes
            
            # Overlay with transparency
            overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        else:
            overlay = image
        
        return overlay

    def create_sequence_visualization(self):
        """Create visualization for the entire sequence"""
        print("Creating sequence visualization...")
        
        # Get all frame IDs
        image_dir = self.preprocessed_path / "image_02"
        frame_files = sorted(list(image_dir.glob("*.png")))
        
        # Create output video
        output_path = self.visualization_path / "sequence_visualization.mp4"
        frame = self.create_frame_visualization(frame_files[0].stem)
        if frame is None:
            raise RuntimeError("Failed to create first frame")
            
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
        
        # Process each frame
        for frame_file in tqdm(frame_files):
            frame = self.create_frame_visualization(frame_file.stem)
            if frame is not None:
                out.write(frame)
        
        out.release()
        print(f"Sequence visualization saved to {output_path}")

def main():
    # Define base path containing all stage outputs
    base_path = "."  # Current directory
    
    # Initialize integrator
    integrator = PipelineIntegrator(base_path)
    
    try:
        # Create 3D visualization
        print("Creating 3D visualization...")
        integrator.create_3d_visualization()
        
        # Create sequence visualization
        print("\nCreating sequence visualization...")
        integrator.create_sequence_visualization()
        
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()