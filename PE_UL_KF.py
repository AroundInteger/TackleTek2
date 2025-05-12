#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Rugby Pose Tracking with Landmark Analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from ultralytics import YOLO
from typing import List, Tuple, Dict
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LandmarkTracker:
    """
    Track landmarks for multiple people with occlusion handling
    """
    def __init__(self, num_landmarks: int, max_people: int = 2):
        """
        Initialize multi-person landmark tracking
        
        Args:
            num_landmarks (int): Number of landmarks per person
            max_people (int): Maximum number of people to track
        """
        self.num_landmarks = num_landmarks
        self.max_people = max_people
        
        # Store landmark data for each person
        self.landmark_histories = [[] for _ in range(max_people)]
        
        # Store association scores between previous and current detections
        self.prev_detections = None
    
    def track_landmarks(self, frame_keypoints: List[np.ndarray]) -> List[np.ndarray]:
        """
        Track landmarks across frames, handling multiple people and potential occlusions
        
        Args:
            frame_keypoints (List[np.ndarray]): Keypoints for each detected person
        
        Returns:
            List[np.ndarray]: Processed landmarks for each person
        """
        # Ensure we don't exceed max people
        frame_keypoints = frame_keypoints[:self.max_people]
        
        # Process landmarks for each person
        processed_landmarks = []
        for person_idx, keypoints in enumerate(frame_keypoints):
            # Validate keypoints (remove (0,0) points)
            valid_keypoints = keypoints[
                (keypoints[:, 0] > 0) | (keypoints[:, 1] > 0)
            ]
            
            # Store current landmarks
            self.landmark_histories[person_idx].append(keypoints)
            
            processed_landmarks.append(keypoints)
        
        return processed_landmarks
    
    def get_landmark_dataframe(self) -> pd.DataFrame:
        """
        Convert landmark tracking history to a pandas DataFrame for analysis
        
        Returns:
            pd.DataFrame: Landmark positions across frames
        """
        # Prepare data for DataFrame
        landmark_data = []
        
        for person_idx, person_history in enumerate(self.landmark_histories):
            if not person_history:
                continue
            
            # Convert history to numpy array for easier processing
            person_landmarks = np.array(person_history)
            
            # Iterate through landmarks
            for landmark_idx in range(self.num_landmarks):
                landmark_frames = person_landmarks[:, landmark_idx]
                
                for frame_idx, landmark in enumerate(landmark_frames):
                    landmark_data.append({
                        'person': person_idx,
                        'landmark_idx': landmark_idx,
                        'frame': frame_idx,
                        'x': landmark[0],
                        'y': landmark[1]
                    })
        
        return pd.DataFrame(landmark_data)

def plot_landmark_trajectories(landmark_df: pd.DataFrame):
    """
    Create visualizations of landmark trajectories
    
    Args:
        landmark_df (pd.DataFrame): Landmark tracking data
    """
    # Create a figure with subplots for x and y trajectories
    plt.figure(figsize=(15, 10))
    
    # X-coordinate trajectories
    plt.subplot(2, 1, 1)
    sns.lineplot(
        data=landmark_df[landmark_df['x'] > 0],
        x='frame', 
        y='x', 
        hue='landmark_idx',
        style='person'
    )
    plt.title('Landmark X-Coordinate Trajectories')
    plt.xlabel('Frame')
    plt.ylabel('X Position')
    
    # Y-coordinate trajectories
    plt.subplot(2, 1, 2)
    sns.lineplot(
        data=landmark_df[landmark_df['y'] > 0],
        x='frame', 
        y='y', 
        hue='landmark_idx',
        style='person'
    )
    plt.title('Landmark Y-Coordinate Trajectories')
    plt.xlabel('Frame')
    plt.ylabel('Y Position')
    
    plt.tight_layout()
    plt.savefig('landmark_trajectories.png')
    plt.close()

def process_video(video_path: str, output_path: str):
    """
    Process video with enhanced landmark tracking and analysis
    """
    # Load YOLO pose estimation model
    try:
        model = YOLO('yolov8x-pose.pt')
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        return
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define landmark connection pairs (body skeleton)
    BODY_CONNECTIONS = [
        (0, 1),   # Head to Neck
        (1, 2), (1, 5),  # Neck to shoulders
        (2, 3), (3, 4),  # Right arm
        (5, 6), (6, 7),  # Left arm
        (1, 8),   # Neck to Hip center
        (8, 9), (9, 10), (8, 11), (11, 12)  # Hip and leg connections
    ]
    
    # Output video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        logging.error(f"Failed to create output video: {e}")
        cap.release()
        return
    
    # Initialize landmark tracker
    tracker = LandmarkTracker(17)  # YOLOv8 provides 17 landmarks
    
    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Pose estimation
            results = model(frame)
            
            # Extract keypoints for all detected people
            if len(results[0].keypoints) > 0:
                # Convert keypoints to numpy arrays
                keypoints_list = [
                    kp.xy[0].cpu().numpy() for kp in results[0].keypoints
                ]
                
                # Track landmarks
                tracked_landmarks = tracker.track_landmarks(keypoints_list)
                
                # Create a copy of the frame for drawing
                annotated_frame = frame.copy()
                
                # Draw landmarks for each person
                for person_landmarks in tracked_landmarks:
                    # Draw landmarks
                    for landmark in person_landmarks:
                        if landmark[0] > 0 or landmark[1] > 0:
                            cv2.circle(annotated_frame, 
                                       (int(landmark[0]), int(landmark[1])), 
                                       5, (0, 255, 0), -1)
                    
                    # Draw landmark connections
                    for start, end in BODY_CONNECTIONS:
                        start_point = person_landmarks[start]
                        end_point = person_landmarks[end]
                        
                        if (start_point[0] > 0 or start_point[1] > 0) and \
                           (end_point[0] > 0 or end_point[1] > 0):
                            cv2.line(annotated_frame, 
                                     (int(start_point[0]), int(start_point[1])), 
                                     (int(end_point[0]), int(end_point[1])), 
                                     (255, 0, 0), 2)
                
                out.write(annotated_frame)
                cv2.imshow('Enhanced Pose Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Generate landmark trajectory plot
    landmark_df = tracker.get_landmark_dataframe()
    plot_landmark_trajectories(landmark_df)
    
    # Save landmark data to CSV
    landmark_df.to_csv('landmark_tracking_data.csv', index=False)
    
    logging.info(f"Processed {frame_count} frames")
    logging.info(f"Output saved to {output_path}")
    logging.info("Landmark trajectory plot saved to landmark_trajectories.png")
    logging.info("Landmark tracking data saved to landmark_tracking_data.csv")

def main():
    # Input and output video paths
    input_video = "R1_1.mp4"
    output_video = "enhanced_pose_tracking.mp4"
    
    process_video(input_video, output_video)

if __name__ == "__main__":
    main()