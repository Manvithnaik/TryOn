import cv2
import numpy as np
import mediapipe as mp
import time

def get_body_measurements():
    """
    Capture a single frame from webcam and extract body measurements
    
    Returns:
        dict: Dictionary containing body measurements (chest, waist, shoulders, height)
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )
    
    # Access webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Set camera resolution to get better results
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Stand in front of the camera with your arms slightly raised to the sides")
    print("Capturing in 3 seconds...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        cap.release()
        return None
    
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Release webcam
    cap.release()
    
    if not results.pose_landmarks:
        print("No pose landmarks detected. Please try again with better lighting and positioning.")
        return None
    
    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    
    # Get image dimensions
    h, w, _ = frame.shape
    
    # Calculate pixel to real-world ratio (assuming height of 1.75m as default)
    # This is a simplification; actual implementation would need calibration
    # Using RIGHT_ANKLE instead of ANKLE_RIGHT (correct MediaPipe landmark name)
    person_height_pixels = abs(landmarks[mp_pose.PoseLandmark.NOSE.value].y - 
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) * h
    pixel_to_meter_ratio = 1.75 / person_height_pixels
    
    # Calculate body measurements
    # Shoulder width
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
    shoulder_width_pixels = np.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                                    (left_shoulder[1] - right_shoulder[1])**2)
    shoulder_width = shoulder_width_pixels * pixel_to_meter_ratio
    
    # Chest width estimation (simplified)
    chest_width_pixels = shoulder_width_pixels * 0.9  # Approximation
    chest_width = chest_width_pixels * pixel_to_meter_ratio
    
    # Waist width estimation (simplified)
    waist_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) * h / 2
    waist_x_left = w * (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x - 0.1)  # Approximation
    waist_x_right = w * (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x + 0.1)  # Approximation
    waist_width_pixels = abs(waist_x_right - waist_x_left)
    waist_width = waist_width_pixels * pixel_to_meter_ratio
    
    # Height
    height = 1.75  # Default assumption, can be changed by user input
    
    # Save the analyzed frame for debugging
    # Draw landmarks on the frame
    mp_drawing = mp.solutions.drawing_utils
    annotated_frame = frame.copy()
    mp_drawing.draw_landmarks(
        annotated_frame, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS
    )
    cv2.imwrite("measurement_frame.jpg", annotated_frame)
    
    return {
        "shoulder_width": shoulder_width,
        "chest_width": chest_width,
        "waist_width": waist_width,
        "height": height
    }