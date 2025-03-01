import cv2
import mediapipe as mp
import csv
import open3d as o3d
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Store pose landmarks for Open3D visualization
landmark_points = []

# Use MediaPipe Pose with default settings
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get pose landmarks
        results = pose.process(rgb_frame)

        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Store landmark points for 3D visualization
            landmark_points = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        # Show the frame with landmarks
        cv2.imshow("Pose Detection", frame)

        # Press 's' to save landmarks ONCE to CSV
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s') and landmark_points:
            with open('pose_landmarks.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Landmark", "X", "Y", "Z"])  # CSV Header
                for i, (x, y, z) in enumerate(landmark_points):
                    writer.writerow([i, x, y, z])
            print("Landmarks saved to CSV!")

        # Press 'q' to exit
        if key == ord('q'):
            break

# Release webcam resources
cap.release()
cv2.destroyAllWindows()

# Display 3D model using Open3D
if landmark_points:
    points = np.array(landmark_points)  # Convert to NumPy array
    pcd = o3d.geometry.PointCloud()  # Create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)  # Assign points
    o3d.visualization.draw_geometries([pcd])  # Show 3D visualization
