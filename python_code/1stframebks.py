import cv2
import mediapipe as mp
import numpy as np

def estimate_background(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    # Initialize the background
    avg_background = np.float32(frame)

    for _ in range(1, num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Accumulate the frames
        cv2.accumulateWeighted(frame, avg_background, 0.1)

    # Convert the accumulated frame to uint8
    background = cv2.convertScaleAbs(avg_background)
    
    cap.release()
    return background

def process_frame_with_background_subtraction(frame, background, min_detection_confidence, min_tracking_confidence):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                        min_tracking_confidence=min_tracking_confidence)

    # Perform background subtraction
    frame_diff = cv2.absdiff(frame, background)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2RGB)

    # Process the image and detect poses
    results = pose.process(frame_rgb)

    # Draw pose annotations on the original frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame

def process_video(input_path, min_detection_confidence, min_tracking_confidence):
    # Estimate background
    background = estimate_background(input_path)
    if background is None:
        return

    # Open the video file again
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Read the first frame (again)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Process the frame
    processed_frame = process_frame_with_background_subtraction(frame, background, min_detection_confidence, min_tracking_confidence)

    # Display the processed frame
    cv2.imshow('Pose Estimation with Background Subtraction', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release resources
    cap.release()

# Example usage
input_video = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/R1.mp4"

# Adjustable parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

process_video(input_video, min_detection_confidence, min_tracking_confidence)
