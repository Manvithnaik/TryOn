import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import collections
import csv

# Initialize Mediapipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Smoothing setup (Moving Average)
history_size = 5
landmark_history = collections.deque(maxlen=history_size)

# Initialize previous landmarks for CSV logging
prev_landmarks = None

# Open CSV file to save pose landmarks
csv_file = open('pose_landmarks.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Landmark", "X", "Y", "Z"])  # CSV Header

def smooth_landmarks(landmarks):
    """Apply moving average smoothing to reduce jitter."""
    landmark_history.append(landmarks)  # Store current landmarks
    return np.mean(landmark_history, axis=0)  # Compute moving average

def visualize_3d_landmarks(landmarks):
    """Visualizes the 3D pose with Open3D."""
    points = np.array(landmarks)

    # Flip 3D pose to correct upside-down issue
    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  
    flipped_landmarks = np.dot(points, rotation_matrix)

    # Define connections between landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
        (0, 5), (5, 6), (6, 7), (7, 8),  # Left arm
        (0, 9), (9, 10), (10, 11), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    ]

    # Create 3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flipped_landmarks)

    # Create 3D skeleton lines
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(flipped_landmarks)
    lines.lines = o3d.utility.Vector2iVector(connections)

    # Assign colors for body parts
    colors = [[1, 0, 0] for _ in range(4)] + [[0, 1, 0] for _ in range(4)] + [[0, 0, 1] for _ in range(4)]
    lines.colors = o3d.utility.Vector3dVector(colors)

    # Load and position 3D clothing model (Shirt)
    try:
        shirt_model = o3d.io.read_triangle_mesh("shirt.obj")  
        shirt_model.compute_vertex_normals()
        shirt_model.paint_uniform_color([0.8, 0.3, 0.3])  # Red color
        shirt_model.translate([0, 0.5, 0])  # Position on shoulders
        o3d.visualization.draw_geometries([pcd, lines, shirt_model], window_name="3D Pose with Try-On")
    except:
        o3d.visualization.draw_geometries([pcd, lines], window_name="3D Pose")

def save_to_csv(landmarks):
    """Saves landmark positions to CSV only if significant movement is detected."""
    global prev_landmarks
    if prev_landmarks is None or np.linalg.norm(landmarks - prev_landmarks) > 0.05:
        for i, landmark in enumerate(landmarks):
            csv_writer.writerow([i, landmark[0], landmark[1], landmark[2]])
        prev_landmarks = landmarks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe Requirement)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose landmarks
    results = pose.process(frame_rgb)

    # Draw pose landmarks on OpenCV window
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmark positions
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

        # Convert to NumPy array
        landmarks = np.array(landmarks)

        # Apply smoothing
        smoothed_landmarks = smooth_landmarks(landmarks)

        # Save to CSV
        save_to_csv(smoothed_landmarks)

        # Real-time Open3D visualization
        visualize_3d_landmarks(smoothed_landmarks)

    # Display Webcam Feed
    cv2.imshow('Webcam Pose', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
    

# Release resources
cap.release()
cv2.destroyAllWindows()
csv_file.close()
