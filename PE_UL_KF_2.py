#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os
import sys
import logging
from collections import deque, defaultdict
import json
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LandmarkTracker:
    def __init__(self, num_landmarks=17):
        self.num_landmarks = num_landmarks
        self.landmark_buffers = defaultdict(list)
        self.kalman_filters = {}
        self.impact_frame = None
        
    def initialize_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        dt = 1.0/30.0  # Time step (assuming 30 fps video)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise (R) - adjusted based on pixel uncertainty
        # Assuming detection uncertainty of ~5 pixels
        kf.R = np.array([[25, 0],    # 5^2 for position variance
                         [0, 25]])
        
        # Process noise (Q) - tuned for human motion
        # Assuming acceleration changes are small between frames
        q = 1.0  # Process noise intensity
        kf.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
        
        # Initial state uncertainty (P)
        kf.P = np.array([
            [10, 0, 0, 0],     # Position uncertainty
            [0, 10, 0, 0],     # Position uncertainty
            [0, 0, 1000, 0],   # Velocity uncertainty
            [0, 0, 0, 1000]    # Velocity uncertainty
        ])
        
        return kf
    
    def process_frame(self, frame_idx, keypoints, player_id):
        """Store landmark positions for a given frame"""
        if keypoints is not None:
            # Convert keypoints to a list of tuples with None for missing landmarks
            processed_keypoints = []
            for i in range(len(keypoints)):
                if np.any(np.isnan(keypoints[i])) or np.all(keypoints[i] == 0):
                    processed_keypoints.append(None)
                else:
                    processed_keypoints.append(tuple(keypoints[i]))
            
            self.landmark_buffers[player_id].append((frame_idx, processed_keypoints))
    
    def set_impact_frame(self, frame_idx):
        """Set the frame where impact occurred"""
        self.impact_frame = frame_idx-3
    
    def interpolate_missing_landmarks(self, player_id):
        """Fill in missing landmarks using Kalman filter predictions"""
        landmarks_data = self.landmark_buffers[player_id]
        
        if player_id not in self.kalman_filters:
            self.kalman_filters[player_id] = [
                self.initialize_kalman_filter() for _ in range(self.num_landmarks)
            ]
        
        interpolated_data = {}
        
        for landmark_idx in range(self.num_landmarks):
            kf = self.kalman_filters[player_id][landmark_idx]
            time_series = []
            
            # Extract time series for this landmark (only valid detections)
            for frame_idx, keypoints in landmarks_data:
                if keypoints[landmark_idx] is not None:
                    time_series.append((frame_idx, keypoints[landmark_idx]))
            
            if not time_series:  # Skip if no valid detections
                continue
                
            # Initialize state with first valid measurement
            first_pos = time_series[0][1]
            kf.x = np.array([first_pos[0], first_pos[1], 0, 0])
            kf.P = np.eye(4) * 100  # Reset covariance
            
            # Process all frames
            current_ts_idx = 0
            last_frame = max(frame_idx for frame_idx, _ in landmarks_data)
            
            for frame_idx in range(last_frame + 1):
                # If we have a valid measurement for this frame
                if current_ts_idx < len(time_series) and time_series[current_ts_idx][0] == frame_idx:
                    pos = time_series[current_ts_idx][1]
                    kf.update(np.array([pos[0], pos[1]]))
                    current_ts_idx += 1
                else:
                    # Just predict if no measurement
                    kf.predict()
                
                # Store interpolated position
                if frame_idx not in interpolated_data:
                    interpolated_data[frame_idx] = [None] * self.num_landmarks
                interpolated_data[frame_idx][landmark_idx] = kf.x[:2]
        
        return interpolated_data

    def plot_landmark_trajectories(self, player_id, landmark_idx, window_size=5):
        """
        Plot original, median filtered, and Kalman filtered trajectories for a specific landmark
        """
        # Get original trajectory data (only valid detections)
        original_data = []
        frames = []
        for frame_idx, keypoints in self.landmark_buffers[player_id]:
            if keypoints[landmark_idx] is not None:  # Only include valid detections
                original_data.append(keypoints[landmark_idx])
                frames.append(frame_idx)
        
        if not original_data:
            logging.warning(f"No valid data available for {player_id} landmark {landmark_idx}")
            return
        
        # Convert to numpy arrays for easier plotting
        original_data = np.array(original_data)
        frames = np.array(frames)
        
        # Apply median filter to original data
        # Pad the data to handle edges
        x_median = np.convolve(original_data[:, 0], 
                              np.ones(window_size)/window_size, 
                              mode='valid')
        y_median = np.convolve(original_data[:, 1], 
                              np.ones(window_size)/window_size, 
                              mode='valid')
        # Adjust frames for median filter (remove edges due to window)
        median_frames = frames[window_size-1:]
        
        # Get Kalman filtered data
        interpolated = self.interpolate_missing_landmarks(player_id)
        filtered_frames = sorted(interpolated.keys())
        filtered_data = []
        for f in filtered_frames:
            if interpolated[f][landmark_idx] is not None:
                filtered_data.append(interpolated[f][landmark_idx])
        filtered_data = np.array(filtered_data)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot X coordinates
        ax1.plot(frames, original_data[:, 0], 'b.', label='Original', alpha=0.5)
        ax1.plot(median_frames, x_median, 'g-', label=f'Median (w={window_size})', alpha=0.7)
        ax1.plot(filtered_frames, filtered_data[:, 0], 'r-', label='Kalman', alpha=0.7)
        ax1.set_title(f'{player_id} Landmark {landmark_idx} - X Coordinate')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('X Position')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Y coordinates
        ax2.plot(frames, original_data[:, 1], 'b.', label='Original', alpha=0.5)
        ax2.plot(median_frames, y_median, 'g-', label=f'Median (w={window_size})', alpha=0.7)
        ax2.plot(filtered_frames, filtered_data[:, 1], 'r-', label='Kalman', alpha=0.7)
        ax2.set_title(f'{player_id} Landmark {landmark_idx} - Y Coordinate')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Y Position')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def calculate_speed(prev_pos, curr_pos, fps):
    """Calculate speed between two positions"""
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) * fps

def identify_players(keypoints, prev_ball_carrier=None, prev_tackler=None):
    """Identify ball carrier and tackler based on position"""
    poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
    
    if len(poses) == 2:
        # Identify based on x-position (left player is ball carrier)
        left_player = min(poses, key=lambda x: x[11, 0])  # Using hip x-coordinate
        right_player = max(poses, key=lambda x: x[11, 0])
        return left_player, right_player
    elif len(poses) == 1 and prev_ball_carrier is not None and prev_tackler is not None:
        # Assign single detection based on distance to previous positions
        pose = poses[0]
        dist_to_ball_carrier = np.linalg.norm(pose[11] - prev_ball_carrier[11])
        dist_to_tackler = np.linalg.norm(pose[11] - prev_tackler[11])
        
        return (pose, prev_tackler) if dist_to_ball_carrier < dist_to_tackler else (prev_ball_carrier, pose)
    
    return None, None

def draw_annotations(frame, ball_carrier, tackler):
    """Draw visual annotations on frame"""
    # Calculate heights
    wrist_height = np.median([ball_carrier[9][1], ball_carrier[10][1]])
    shoulder_height = np.median([tackler[5][1], tackler[6][1]])
    
    # Draw markers for key points
    # Ball carrier's wrists
    cv2.circle(frame, tuple(map(int, ball_carrier[9])), 5, (0, 255, 0), -1)  # Left wrist
    cv2.circle(frame, tuple(map(int, ball_carrier[10])), 5, (0, 255, 0), -1)  # Right wrist
    
    # Tackler's head and shoulders
    cv2.circle(frame, tuple(map(int, tackler[0])), 5, (255, 0, 0), -1)  # Head
    cv2.circle(frame, tuple(map(int, tackler[5])), 5, (255, 0, 0), -1)  # Left shoulder
    cv2.circle(frame, tuple(map(int, tackler[6])), 5, (255, 0, 0), -1)  # Right shoulder
    
    # Calculate and display height difference
    height_difference = shoulder_height - wrist_height
    color = (0, 255, 0) if height_difference > 0 else (0, 0, 255)
    cv2.putText(frame, f"Shoulder-Wrist Diff: {height_difference:.2f}",
                (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return height_difference, shoulder_height, wrist_height

def detect_impact(ball_carrier, tackler, threshold=50):
    """Detect tackle impact based on proximity"""
    # Calculate distance between players' torsos
    ball_carrier_torso = np.mean(ball_carrier[5:13], axis=0)  # Average of shoulders and hips
    tackler_torso = np.mean(tackler[5:13], axis=0)
    
    distance = np.linalg.norm(ball_carrier_torso - tackler_torso)
    return distance < threshold

def main():
    # Initialize video capture and YOLO model
    video_path = "R1_1.mp4"
    cap = cv2.VideoCapture(video_path)
    model = YOLO('yolov8x-pose.pt')
    
    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        sys.exit(1)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize output video writer
    output_path = 'analyzed_rugby_tackle.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize landmark tracker
    tracker = LandmarkTracker()
    
    # Prepare CSV file
    csv_filename = 'tackle_analysis.csv'
    csv_header = ['frame', 'ball_carrier_speed', 'tackler_speed', 
                 'shoulder_wrist_difference', 'tackler_shoulder_height', 'ball_carrier_wrist_height']
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)
        
        # Process frames until impact
        frame_count = 0
        impact_detected = False
        prev_ball_carrier = None
        prev_tackler = None
        
        while cap.isOpened() and not impact_detected:
            success, frame = cap.read()
            if not success:
                break
            
            # Run pose estimation
            results = model(frame)
            annotated_frame = frame.copy()
            
            if len(results[0].keypoints) >= 1:
                # Identify players
                ball_carrier, tackler = identify_players(
                    results[0].keypoints, prev_ball_carrier, prev_tackler)
                
                if ball_carrier is not None and tackler is not None:
                    # Store positions
                    prev_ball_carrier, prev_tackler = ball_carrier, tackler
                    tracker.process_frame(frame_count, ball_carrier, 'ball_carrier')
                    tracker.process_frame(frame_count, tackler, 'tackler')
                    
                    # Draw annotations
                    height_diff, shoulder_height, wrist_height = draw_annotations(
                        annotated_frame, ball_carrier, tackler)
                    
                    # Calculate speeds (if we have previous positions)
                    speeds = [0, 0]
                    if frame_count > 0:
                        speeds = [
                            calculate_speed(prev_ball_carrier[0], ball_carrier[0], fps),
                            calculate_speed(prev_tackler[0], tackler[0], fps)
                        ]
                    
                    # Write to CSV
                    csv_writer.writerow([
                        frame_count, speeds[0], speeds[1],
                        height_diff, shoulder_height, wrist_height
                    ])
                    
                    # Check for impact
                    impact_detected = detect_impact(ball_carrier, tackler)
                    if impact_detected:
                        tracker.set_impact_frame(frame_count)
                        logging.info(f"Impact detected at frame {frame_count}")
            
            # Write frame
            out.write(annotated_frame)
            cv2.imshow("Rugby Analysis", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            frame_count += 1
        
        # After impact, interpolate missing landmarks and plot
        if impact_detected:
            window_size = 3  # or any other value
            tracker.plot_landmark_trajectories('tackler', 6, window_size)
            # Plot trajectories for key landmarks
            # Ball carrier's wrists
            tracker.plot_landmark_trajectories('ball_carrier', 9, window_size)  # Left wrist
            tracker.plot_landmark_trajectories('ball_carrier', 10, window_size)  # Right wrist
            
            # Tackler's head and shoulders
            tracker.plot_landmark_trajectories('tackler', 0, window_size)  # Head
            tracker.plot_landmark_trajectories('tackler', 5, window_size)  # Left shoulder
            #tracker.plot_landmark_trajectories('tackler', 6)  # Right shoulder
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    logging.info(f"Analysis complete. Results saved to {csv_filename}")

if __name__ == "__main__":
    main()