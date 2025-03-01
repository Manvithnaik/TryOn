import cv2
import numpy as np
from pose_tracker import PoseTracker
from model_loader import ModelLoader
from visualizer import Visualizer

class VirtualTryOn:
    def __init__(self):
        self.pose_tracker = PoseTracker()
        self.model_loader = ModelLoader()
        self.visualizer = Visualizer(self.model_loader.body_model, self.model_loader.shirt_model)
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            landmarks = self.pose_tracker.track_pose(frame)

            if landmarks:
                mid_shoulder = np.array([
                    (landmarks.landmark[11].x + landmarks.landmark[12].x) / 2,
                    (landmarks.landmark[11].y + landmarks.landmark[12].y) / 2,
                    (landmarks.landmark[11].z + landmarks.landmark[12].z) / 2
                ])

                scale_factor = np.linalg.norm(
                    np.array([landmarks.landmark[11].x, landmarks.landmark[11].y]) -
                    np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])
                ) * 5  # Increase scaling to make models more visible

                # Debugging print statements
                print(f"Updating models - Position: {mid_shoulder}, Scale: {scale_factor:.2f}")

                # Update Open3D models
                self.visualizer.update_models(mid_shoulder, scale_factor)
                self.visualizer.force_refresh()  # Fix disappearing models

            cv2.imshow("Virtual Try-On", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        self.visualizer.close()

if __name__ == "__main__":
    tryon = VirtualTryOn()
    tryon.run()
