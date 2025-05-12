#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:09:15 2024

@author: mrbimac
"""

import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os
import logging
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')  # Using the most accurate model

def calculate_speed(prev_pos, curr_pos, fps):
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) * fps

def identify_players(keypoints, prev_ball_carrier, prev_tackler):
    poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
    
    if len(poses) == 2:
        # If we have two poses, identify based on hip position
        left_player = min(poses, key=lambda x: x[11, 0])  # Left hip x-coordinate
        right_player = max(poses, key=lambda x: x[11, 0])  # Left hip x-coordinate
        return left_player, right_player
    elif len(poses) == 1:
        # If we have only one pose, assign it based on minimum distance to previous positions
        pose = poses[0]
        dist_to_ball_carrier = np.linalg.norm(pose[11] - prev_ball_carrier[11])
        dist_to_tackler = np.linalg.norm(pose[11] - prev_tackler[11])
        
        if dist_to_ball_carrier < dist_to_tackler:
            return pose, prev_tackler
        else:
            return prev_ball_carrier, pose
    else:
        # If we don't have any poses, return the previous positions
        return prev_ball_carrier, prev_tackler

def initialize_players(keypoints):
    poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
    if len(poses) >= 2:
        left_player = min(poses, key=lambda x: x[11, 0])  # Left hip x-coordinate
        right_player = max(poses, key=lambda x: x[11, 0])  # Left hip x-coordinate
        return left_player, right_player
    else:
        return None, None

def draw_square(frame, center, size, color, thickness=-1):
    half_size = size // 2
    top_left = (int(center[0]) - half_size, int(center[1]) - half_size)
    bottom_right = (int(center[0]) + half_size, int(center[1]) + half_size)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

def draw_annotations(frame, ball_carrier, tackler, ball_carrier_history, tackler_history):
    # Define distinct colors
    colors = {
        'ball_carrier_wrist': (0, 255, 0),    # Green
        'ball_carrier_hip': (255, 0, 255),    # Magenta
        'ball_carrier_knee': (255, 255, 0),   # Yellow
        'tackler_head': (255, 0, 0),          # Blue
        'tackler_shoulder': (0, 165, 255),    # Orange
        'tackler_hip': (128, 0, 128)          # Purple
    }

    # Draw historical positions for ball carrier
    for pos in ball_carrier_history:
        draw_square(frame, pos[9], 3, colors['ball_carrier_wrist'])   # Left wrist
        draw_square(frame, pos[10], 3, colors['ball_carrier_wrist'])  # Right wrist
        draw_square(frame, pos[11], 3, colors['ball_carrier_hip'])    # Left hip
        draw_square(frame, pos[12], 3, colors['ball_carrier_hip'])    # Right hip
        draw_square(frame, pos[13], 3, colors['ball_carrier_knee'])   # Left knee
        draw_square(frame, pos[14], 3, colors['ball_carrier_knee'])   # Right knee

    # Draw historical positions for tackler
    for pos in tackler_history:
        draw_square(frame, pos[0], 3, colors['tackler_head'])        # Head
        draw_square(frame, pos[5], 3, colors['tackler_shoulder'])    # Left shoulder
        draw_square(frame, pos[6], 3, colors['tackler_shoulder'])    # Right shoulder
        draw_square(frame, pos[11], 3, colors['tackler_hip'])        # Left hip
        draw_square(frame, pos[12], 3, colors['tackler_hip'])        # Right hip

    # Draw current positions with larger markers
    draw_square(frame, ball_carrier[9], 8, colors['ball_carrier_wrist'])   # Left wrist
    draw_square(frame, ball_carrier[10], 8, colors['ball_carrier_wrist'])  # Right wrist
    draw_square(frame, tackler[0], 8, colors['tackler_head'])              # Head

    # Calculate heights
    wrist_height = np.median([ball_carrier[9][1], ball_carrier[10][1]])
    shoulder_height = np.median([tackler[5][1], tackler[6][1]])
    
    # Calculate shoulder-wrist height difference
    height_difference = shoulder_height - wrist_height
    
    # Determine color based on height difference
    diff_color = (0, 255, 0) if height_difference < 0 else (0, 0, 255)
    
    # Display height difference
    cv2.putText(frame, f"Shoulder-Wrist Diff: {height_difference:.2f}", 
                (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_color, 2)

    return height_difference, shoulder_height, np.median([ball_carrier[11][1], ball_carrier[12][1]])

def detect_impact(ball_carrier, tackler):
    # Calculate median wrist position for ball carrier
    ball_carrier_wrist_x = np.median([ball_carrier[9][0], ball_carrier[10][0]])
    
    # Calculate median shoulder position for tackler
    tackler_shoulder_x = np.median([tackler[5][0], tackler[6][0]])
    
    # Check if the x-direction distance is <= 0
    return ball_carrier_wrist_x - tackler_shoulder_x <= 0

# Open the video file
video_path = "R1_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare output video
output_path = 'analyzed_rugby_tackle_R1_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Prepare CSV file
csv_filename = 'tackle_analysis_R1_1.csv'
csv_header = ['frame', 'ball_carrier_speed', 'tackler_speed', 'shoulder_wrist_height_difference', 'tackler_shoulder_height', 'ball_carrier_waist_height']

ball_carrier_positions = []
tackler_positions = []
ball_carrier_history = deque(maxlen=30)  # Adjust maxlen to change the length of the displayed history
tackler_history = deque(maxlen=30)
impact_detected = False
prev_ball_carrier = None
prev_tackler = None

try:
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        frame_count = 0
        while cap.isOpened() and not impact_detected:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            annotated_frame = results[0].plot()  # Re-enable Ultralytics pose estimation

            if len(results[0].keypoints) >= 1:
                if prev_ball_carrier is None or prev_tackler is None:
                    # Initialize players for the first frame with detections
                    ball_carrier, tackler = initialize_players(results[0].keypoints)
                else:
                    # Use simplified identification for subsequent frames
                    ball_carrier, tackler = identify_players(results[0].keypoints, prev_ball_carrier, prev_tackler)

                if ball_carrier is not None and tackler is not None:
                    prev_ball_carrier, prev_tackler = ball_carrier, tackler

                    ball_carrier_positions.append(ball_carrier[11])  # Using left hip point (index 11)
                    tackler_positions.append(tackler[11])
                    
                    ball_carrier_history.append(ball_carrier)
                    tackler_history.append(tackler)
                    
                    if len(ball_carrier_positions) > 2:
                        ball_carrier_positions.pop(0)
                        tackler_positions.pop(0)

                    # Calculate speeds
                    ball_carrier_speed = 0
                    tackler_speed = 0
                    if len(ball_carrier_positions) > 1:
                        ball_carrier_speed = calculate_speed(ball_carrier_positions[-2], ball_carrier_positions[-1], fps)
                        tackler_speed = calculate_speed(tackler_positions[-2], tackler_positions[-1], fps)

                    # Draw additional annotations
                    height_difference, shoulder_height, waist_height = draw_annotations(
                        annotated_frame, ball_carrier, tackler, ball_carrier_history, tackler_history)

                    # Write data to CSV
                    csv_writer.writerow([frame_count, ball_carrier_speed, tackler_speed, height_difference, shoulder_height, waist_height])

                    # Check for impact
                    impact_detected = detect_impact(ball_carrier, tackler)

                    if impact_detected:
                        logging.info(f"Impact detected at frame {frame_count}")
                else:
                    logging.info(f"Players lost at frame {frame_count}")
            else:
                logging.info(f"No players detected at frame {frame_count}")

            out.write(annotated_frame)
            cv2.imshow("Rugby Tackle Analysis", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

        # If impact is detected, show the final frame with annotations for a few seconds
        if impact_detected and prev_ball_carrier is not None and prev_tackler is not None:
            final_frame = annotated_frame.copy()
            draw_annotations(final_frame, prev_ball_carrier, prev_tackler, ball_carrier_history, tackler_history)
            for _ in range(fps * 3):  # Show for 3 seconds
                out.write(final_frame)

finally:
    # Ensure resources are released
    cap.release()
    out.release()
    cv2.destroyAllWindows()

logging.info(f"Analysis complete. Results saved to {csv_filename} and {output_path}")

# Verify if the output video was created
if os.path.exists(output_path):
    logging.info(f"Output video successfully saved to {output_path}")
    logging.info(f"File size: {os.path.getsize(output_path)} bytes")
else:
    logging.error(f"Failed to save output video to {output_path}")