import cv2
import mediapipe as mp
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Kalman Filter for smoother tracking
        self.kf = self.init_kalman_filter()

    def init_kalman_filter(self):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([0, 0, 0, 0, 0, 0])
        dt = 1.0 / 30.0
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

    def track_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            return results.pose_landmarks

        return None
