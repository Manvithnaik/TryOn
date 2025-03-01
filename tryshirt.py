import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import os
import copy

# Initialize Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load 3D Models with Error Handling
script_dir = os.path.dirname(os.path.abspath(__file__))
shirt_path = os.path.join(script_dir, "shirt.obj")
body_path = os.path.join(script_dir, "body.obj")

def load_mesh(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found. Check file path and format.")
        return None
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        print(f"Error: Unable to load {file_path}. Ensure the file is not corrupted.")
        return None
    mesh.compute_vertex_normals()
    return mesh

shirt_mesh = load_mesh(shirt_path)
body_mesh = load_mesh(body_path)

if shirt_mesh is None or body_mesh is None:
    exit()

# Start Video Capture
cap = cv2.VideoCapture(0)

# Exponential Moving Average (EMA) for smoothing
ema_alpha = 0.7  # Adjust smoothness factor
previous_translation = None
previous_scale = None
previous_rotation = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract keypoints for alignment
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        
        # Compute translation (align shirt to mid-shoulder)
        translation = mid_shoulder - np.mean(np.asarray(shirt_mesh.vertices), axis=0)
        if previous_translation is not None:
            translation = ema_alpha * translation + (1 - ema_alpha) * previous_translation
        previous_translation = translation
        
        # Compute scaling (based on shoulder width)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        vertices_array = np.asarray(shirt_mesh.vertices)
        if len(vertices_array) < 2:
            print("Error: Not enough vertices in the shirt mesh.")
            continue
        shirt_width = np.linalg.norm(vertices_array[0] - vertices_array[1])
        scale_factor = shoulder_width / shirt_width
        if previous_scale is not None:
            scale_factor = ema_alpha * scale_factor + (1 - ema_alpha) * previous_scale
        previous_scale = scale_factor
        
        # Compute rotation using quaternions
        shoulder_vector = right_shoulder - left_shoulder
        angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -angle))
        if previous_rotation is not None:
            rotation_matrix = ema_alpha * rotation_matrix + (1 - ema_alpha) * previous_rotation
        previous_rotation = rotation_matrix
        
        # Apply transformations to copies of the meshes
        transformed_shirt = copy.deepcopy(shirt_mesh)
        transformed_body = copy.deepcopy(body_mesh)
        
        transformed_shirt.translate(translation, relative=False)
        transformed_shirt.scale(scale_factor, center=np.mean(vertices_array, axis=0))
        transformed_shirt.rotate(rotation_matrix, center=mid_shoulder)
        
        transformed_body.translate(translation, relative=False)
        transformed_body.scale(scale_factor, center=np.mean(np.asarray(body_mesh.vertices), axis=0))
        transformed_body.rotate(rotation_matrix, center=mid_shoulder)
        
        # Render updated meshes
        o3d.visualization.draw_geometries([transformed_body, transformed_shirt])
    
    # Display video feed
    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()