#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 2024
@author: rowanbrown
Integrated Rugby Analysis System combining background subtraction, 
grid visualization, and pose estimation
"""

import cv2
import numpy as np
import logging
import os
import sys
from collections import defaultdict, deque
from ultralytics import YOLO
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BackgroundSubtractor:
    def __init__(self, history=1000):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=50,
            detectShadows=False
        )
        self.background_model = None
        self.frames_processed = 0
        
    def update(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        self.frames_processed += 1
        
        if self.frames_processed >= 100:
            self.background_model = self.bg_subtractor.getBackgroundImage()
        
        return fg_mask, self.background_model

class PoseTracker:
    def __init__(self, model_path='yolov8x-pose.pt'):
        self.model = YOLO(model_path)
        self.ball_carrier_history = deque(maxlen=1)
        self.tackler_history = deque(maxlen=1)
        self.prev_ball_carrier = None
        self.prev_tackler = None
        self.impact_detected = False
        
    def identify_players(self, keypoints):
        # Convert keypoints to numpy array of xy coordinates
        poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
        
        if len(poses) == 2:
            left_player = min(poses, key=lambda x: x[11, 0])
            right_player = max(poses, key=lambda x: x[11, 0])
            return left_player, right_player
        elif len(poses) == 1:
            pose = poses[0]
            if self.prev_ball_carrier is not None and self.prev_tackler is not None:
                dist_to_carrier = np.linalg.norm(pose[11] - self.prev_ball_carrier[11])
                dist_to_tackler = np.linalg.norm(pose[11] - self.prev_tackler[11])
                return (pose, self.prev_tackler) if dist_to_carrier < dist_to_tackler else (self.prev_ball_carrier, pose)
        return self.prev_ball_carrier, self.prev_tackler

    def update(self, frame):
        # Run pose detection on the frame
        results = self.model(frame)
        
        # Check if any poses are detected
        if len(results[0].keypoints) >= 1:
            # Identify players based on keypoints
            ball_carrier, tackler = self.identify_players(results[0].keypoints)
            
            # If both ball carrier and tackler are identified
            if ball_carrier is not None and tackler is not None:
                # Update previous ball carrier and tackler
                self.prev_ball_carrier, self.prev_tackler = ball_carrier, tackler
                
                # Add to history
                self.ball_carrier_history.append(ball_carrier)
                self.tackler_history.append(tackler)
                
                # Check for impact
                self.impact_detected = self.detect_impact(ball_carrier, tackler)
                
                # Return whether players were detected and the annotated frame
                return True, results[0].plot()
        
        # If no players detected
        return False, frame

    def detect_impact(self, ball_carrier, tackler):
        # Check proximity of ball carrier's wrist to tackler's shoulder
        ball_carrier_wrist_x = np.median([ball_carrier[9][0], ball_carrier[10][0]])
        tackler_shoulder_x = np.median([tackler[5][0], tackler[6][0]])
        return abs(ball_carrier_wrist_x - tackler_shoulder_x) <= 10

    def draw_annotations(self, frame):
        if self.prev_ball_carrier is None or self.prev_tackler is None:
            return frame

        colors = {
            'ball_carrier': (0, 255, 0),    # Green
            'tackler': (0, 165, 255)        # Orange
        }

        # Draw historical positions
        for hist_pos in self.ball_carrier_history:
            frame = self._draw_skeleton(frame, hist_pos, colors['ball_carrier'], 1)
        
        for hist_pos in self.tackler_history:
            frame = self._draw_skeleton(frame, hist_pos, colors['tackler'], 1)

        # Draw current positions
        frame = self._draw_skeleton(frame, self.prev_ball_carrier, colors['ball_carrier'], 2)
        frame = self._draw_skeleton(frame, self.prev_tackler, colors['tackler'], 2)

        return frame

    def _draw_skeleton(self, frame, pose_data, color, thickness):
        """
        Draw skeleton for a given pose.
        
        Args:
            frame (numpy.ndarray): The image to draw on
            pose_data (numpy.ndarray or list): Keypoints of the pose
            color (tuple): Color for drawing the skeleton
            thickness (int): Thickness of skeleton lines
        """
        # Define skeleton connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head to spine
            (0, 4), (4, 5), (5, 6),          # Left arm
            (0, 8), (8, 9), (9, 10),         # Right arm
            (7, 11), (11, 12),               # Hips
            (11, 13), (13, 15),              # Left leg
            (12, 14), (14, 16)               # Right leg
        ]
        
        # Convert to numpy array if it's not already
        pose = np.array(pose_data)
        
        # Draw connections
        for start_idx, end_idx in connections:
            # Check if indices are valid
            if (0 <= start_idx < len(pose) and 0 <= end_idx < len(pose)):
                start_point = tuple(map(int, pose[start_idx]))
                end_point = tuple(map(int, pose[end_idx]))
                
                # Additional check to avoid drawing lines to origin
                if (start_point[0] > 1 and start_point[1] > 1 and 
                    end_point[0] > 1 and end_point[1] > 1):
                    cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw joints (with origin check)
        for point in pose:
            point_coords = tuple(map(int, point))
            if point_coords[0] > 1 and point_coords[1] > 1:
                cv2.circle(frame, point_coords, 3, color, -1)
        
        return frame

def main():
    video_path = "R1_1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize components
    bg_subtractor = BackgroundSubtractor()
    pose_tracker = PoseTracker()
    
    # Set up video writer
    output_path = 'rugby_integrated_analysis.mp4'
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (width, height * 2))

    # Set up CSV logging
    csv_filename = 'rugby_integrated_analysis.csv'
    csv_header = ['frame', 'impact_detected', 'players_detected']
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)
            
            frame_count = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break

                # Update pose tracking
                players_detected, pose_frame = pose_tracker.update(frame)
                
                # Update background subtraction
                fg_mask, bg_model = bg_subtractor.update(frame)
                
                # Create visualization
                if bg_model is not None:
                    debug_view = np.vstack([
                        pose_frame,
                        cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                    ])
                else:
                    debug_view = np.vstack([
                        pose_frame,
                        np.zeros_like(frame)
                    ])

                # Add annotations
                pose_tracker.draw_annotations(debug_view)
                
                # Write data to CSV
                csv_writer.writerow([
                    frame_count,
                    pose_tracker.impact_detected,
                    players_detected
                ])

                # Write frame to video
                out.write(debug_view)
                
                # Display
                cv2.imshow('Rugby Analysis', debug_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Verify output
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Analysis video saved successfully to {output_path}")
    else:
        logging.error(f"The output video file {output_path} was not created or is empty")

if __name__ == "__main__":
    main()