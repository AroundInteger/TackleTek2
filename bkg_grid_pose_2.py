import cv2
import numpy as np
import logging
import os
import sys
from collections import defaultdict, deque
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt

class LandmarkTracker:
    def __init__(self, model_path='yolov8x-pose.pt'):
        self.model = YOLO(model_path)
        self.landmark_history = defaultdict(list)
        self.frame_numbers = []
        
    def update(self, frame, frame_number):
        # Run pose detection on the frame
        results = self.model(frame)
        
        # Check if any poses are detected
        if len(results[0].keypoints) >= 1:
            # Convert keypoints to numpy array of xy coordinates
            poses = [kp.xy[0].cpu().numpy() for kp in results[0].keypoints]
            
            # Store frame number
            self.frame_numbers.append(frame_number)
            
            # Store each landmark for each detected pose
            for pose_idx, pose in enumerate(poses):
                for landmark_idx, landmark in enumerate(pose):
                    self.landmark_history[f'Pose_{pose_idx}_Landmark_{landmark_idx}'].append(landmark)
        
        return len(results[0].keypoints) > 0
    
    def plot_landmarks(self):
        # Ensure matplotlib is in non-interactive mode
        plt.ioff()
        
        # Create a figure with subplots for x and y coordinates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Player Landmark Positions Over Time')
        
        # Landmark labels for better understanding
        landmark_labels = [
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 
            'Left Shoulder', 'Right Shoulder', 'Neck', 
            'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
            'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 
            'Left Ankle', 'Right Ankle'
        ]
        
        # Plot x coordinates
        for landmark_name, landmark_values in self.landmark_history.items():
            # Extract pose and landmark index
            pose_idx = int(landmark_name.split('_')[1])
            landmark_idx = int(landmark_name.split('_')[3])
            
            # Use more informative label
            display_name = f'Pose_{pose_idx}_{landmark_labels[landmark_idx]}'
            
            # Truncate to the minimum length to ensure consistent plotting
            plot_length = min(len(self.frame_numbers), len(landmark_values))
            x_coords = [point[0] for point in landmark_values[:plot_length]]
            ax1.plot(self.frame_numbers[:plot_length], x_coords, label=display_name)
        
        ax1.set_title('X Coordinate of Landmarks')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('X Position')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Plot y coordinates
        for landmark_name, landmark_values in self.landmark_history.items():
            # Extract pose and landmark index
            pose_idx = int(landmark_name.split('_')[1])
            landmark_idx = int(landmark_name.split('_')[3])
            
            # Use more informative label
            display_name = f'Pose_{pose_idx}_{landmark_labels[landmark_idx]}'
            
            # Truncate to the minimum length to ensure consistent plotting
            plot_length = min(len(self.frame_numbers), len(landmark_values))
            y_coords = [point[1] for point in landmark_values[:plot_length]]
            ax2.plot(self.frame_numbers[:plot_length], y_coords, label=display_name)
        
        ax2.set_title('Y Coordinate of Landmarks')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Y Position')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        
        plt.savefig('landmark_tracking.png', dpi=300)
        plt.close()

def main():
    video_path = "R1_1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        sys.exit(1)

    # Initialize landmark tracker
    landmark_tracker = LandmarkTracker()
    
    try:
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Update landmark tracking
            landmark_tracker.update(frame, frame_count)
            
            frame_count += 1
            
            # Optional: Break after processing a subset of frames for testing
            if frame_count > 1000:
                break

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

    finally:
        cap.release()

    # Plot landmarks
    landmark_tracker.plot_landmarks()
    logging.info("Landmark tracking plot saved as landmark_tracking.png")

if __name__ == "__main__":
    main()