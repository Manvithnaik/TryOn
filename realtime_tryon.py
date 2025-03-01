import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import threading
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class VirtualTryOn:
    def __init__(self):
        # Constants
        self.BASE_SHOULDER_WIDTH = 0.3  # Base shoulder width in meters
        self.SMOOTHING_FACTOR = 0.7     # For position smoothing (0-1)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize OpenCV
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize 3D visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600)
        
        # Load 3D models
        self.body_mesh = self.load_and_process_mesh("models/body.obj")
        self.shirt_mesh = self.load_and_process_mesh("models/shirt.obj", is_shirt=True)
        
        # Store original vertices for resetting transformations
        self.original_body_vertices = np.asarray(self.body_mesh.vertices).copy()
        self.original_shirt_vertices = np.asarray(self.shirt_mesh.vertices).copy()
        
        # Add models to the visualizer
        self.vis.add_geometry(self.body_mesh)
        self.vis.add_geometry(self.shirt_mesh)
        
        # Set up the view
        self.setup_camera_view()
        
        # Initialize tracking variables
        self.last_shoulder_pos = np.array([0, 0, 0])
        self.last_shoulder_width = self.BASE_SHOULDER_WIDTH
        self.running = True
        
        # Initialize Kalman filter for position tracking
        self.kf = self.init_kalman_filter()
        
        # For thread synchronization
        self.lock = threading.Lock()
    
    def process_frame(self, frame):
        """Process a single frame to detect pose and update the model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        
        return frame
    
    def setup_camera_view(self):
        """Set up Open3D camera view for proper visualization"""
        view_control = self.vis.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        
        # Set camera position
        cam_params.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 3],  # Move camera 3 units away
            [0, 0, 0, 1]
        ])
        
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        self.vis.update_renderer()
    
    def init_kalman_filter(self):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([0, 0, 0, 0, 0, 0])
        dt = 1.0/30.0  # Assuming 30 fps
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        kf.R = np.eye(3) * 0.01
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05, block_size=3)
        kf.P = np.eye(6) * 1.0
        return kf
    
    def cleanup(self):
        """Ensure Open3D and OpenCV resources are released properly"""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        try:
            self.vis.destroy_window()
        except:
            pass
        print("Application terminated successfully")
    
    def run(self):
        """Run in single-threaded mode to avoid OpenGL context issues"""
        try:
            print("Running in single-threaded mode...")
            while self.running and self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read from webcam")
                    break
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_frame(frame)
                cv2.imshow('Virtual Try-On', processed_frame)
                self.vis.poll_events()
                self.vis.update_renderer()
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    self.running = False
                    break
        except KeyboardInterrupt:
            print("Application stopped by user")
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        print("Starting Virtual Try-On System...")
        print("Press 'q' to exit")
        tryon = VirtualTryOn()
        tryon.run()
    except Exception as e:
        print(f"An error occurred: {e}")
