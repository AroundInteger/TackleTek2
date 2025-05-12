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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')  # Using the most accurate model

def calculate_speed(prev_pos, curr_pos, fps):
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) * fps

def identify_players_initial(keypoints, num_frames=10):
    # Identify players based on their x-position for the first num_frames
    sorted_keypoints = sorted(keypoints, key=lambda x: x.xy[0, 0, 0])
    return sorted_keypoints[0].xy[0].cpu().numpy(), sorted_keypoints[1].xy[0].cpu().numpy()

def identify_players(keypoints, prev_ball_carrier, prev_tackler):
    if len(keypoints) < 2:
        return None, None

    poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
    
    # Calculate the mean squared error between current poses and previous poses
    mse_ball_carrier = [np.mean((pose - prev_ball_carrier)**2) for pose in poses]
    mse_tackler = [np.mean((pose - prev_tackler)**2) for pose in poses]
    
    # Identify players based on minimum MSE
    ball_carrier_idx = np.argmin(mse_ball_carrier)
    tackler_idx = np.argmin(mse_tackler)
    
    # Ensure we don't assign the same pose to both players
    if ball_carrier_idx == tackler_idx:
        if mse_ball_carrier[ball_carrier_idx] < mse_tackler[tackler_idx]:
            tackler_idx = 1 - ball_carrier_idx
        else:
            ball_carrier_idx = 1 - tackler_idx
    
    return poses[ball_carrier_idx], poses[tackler_idx]

def draw_annotations(frame, ball_carrier, tackler, head_positions, shoulder_positions, waist_positions):
    # Calculate median wrist height for ball carrier
    wrist_height = np.median([ball_carrier[9][1], ball_carrier[10][1]])
    
    # Calculate median shoulder height for tackler (using left and right shoulders)
    shoulder_height = np.median([tackler[5][1], tackler[6][1]])
    
    # Calculate waist height for ball carrier (using left and right hips)
    waist_height = np.median([ball_carrier[11][1], ball_carrier[12][1]])
    
    # Add current positions
    head_positions.append((int(tackler[0][0]), int(tackler[0][1])))  # Using actual head position
    shoulder_positions.append((int(tackler[5][0]), int(shoulder_height)))
    waist_positions.append((int(ball_carrier[11][0]), int(waist_height)))
    
    # Draw current positions
    cv2.circle(frame, (int(ball_carrier[9][0]), int(wrist_height)), 5, (0, 255, 0), -1)  # Green for ball carrier's wrist
    cv2.circle(frame, (int(tackler[0][0]), int(tackler[0][1])), 5, (255, 0, 0), -1)  # Blue for tackler's head
    cv2.circle(frame, (int(tackler[5][0]), int(shoulder_height)), 5, (0, 165, 255), -1)  # Orange for tackler's shoulders
    cv2.circle(frame, (int(ball_carrier[11][0]), int(waist_height)), 5, (255, 0, 255), -1)  # Magenta for ball carrier's waist
    
    # Draw historical positions
    for pos in head_positions[:-1]:
        cv2.circle(frame, pos, 3, (255, 0, 0), -1)  # Blue dot for tackler's head
    for pos in shoulder_positions[:-1]:
        cv2.circle(frame, pos, 3, (0, 165, 255), -1)  # Orange dot for tackler's shoulders
    for pos in waist_positions[:-1]:
        cv2.circle(frame, pos, 3, (255, 0, 255), -1)  # Magenta dot for ball carrier's waist
    
    # Calculate shoulder-wrist height difference
    height_difference = shoulder_height - wrist_height
    
    # Determine color based on height difference
    color = (0, 255, 0) if height_difference > 0 else (0, 0, 255)
    
    # Display height difference
    cv2.putText(frame, f"Shoulder-Wrist Diff: {height_difference:.2f}", 
                (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return height_difference, shoulder_height, waist_height

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
head_positions = []
shoulder_positions = []
waist_positions = []
impact_detected = False
prev_ball_carrier = None
prev_tackler = None

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    frame_count = 0
    while cap.isOpened() and not impact_detected:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        
        # Instead of using results[0].plot(), we'll draw our own annotations
        annotated_frame = frame.copy()

        if len(results[0].keypoints) >= 2:
            if frame_count < 10:
                # Use initial identification for the first 10 frames
                ball_carrier, tackler = identify_players_initial(results[0].keypoints)
            else:
                # Use landmark-based tracking for subsequent frames
                ball_carrier, tackler = identify_players(results[0].keypoints, prev_ball_carrier, prev_tackler)

            if ball_carrier is not None and tackler is not None:
                prev_ball_carrier, prev_tackler = ball_carrier, tackler

                ball_carrier_positions.append(ball_carrier[0])  # Using hip point (index 0)
                tackler_positions.append(tackler[0])
                
                if len(ball_carrier_positions) > 2:
                    ball_carrier_positions.pop(0)
                    tackler_positions.pop(0)

                # Calculate speeds
                ball_carrier_speed = 0
                tackler_speed = 0
                if len(ball_carrier_positions) > 1:
                    ball_carrier_speed = calculate_speed(ball_carrier_positions[-2], ball_carrier_positions[-1], fps)
                    tackler_speed = calculate_speed(tackler_positions[-2], tackler_positions[-1], fps)

                # Draw annotations and get height differences
                height_difference, shoulder_height, waist_height = draw_annotations(
                    annotated_frame, ball_carrier, tackler, head_positions, shoulder_positions, waist_positions)

                # Write data to CSV
                csv_writer.writerow([frame_count, ball_carrier_speed, tackler_speed, height_difference, shoulder_height, waist_height])
            else:
                impact_detected = True
                logging.info(f"Impact detected at frame {frame_count}")
        else:
            impact_detected = True
            logging.info(f"Impact detected at frame {frame_count}")

        out.write(annotated_frame)
        cv2.imshow("Rugby Tackle Analysis", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # If impact is detected, show the final frame for a few seconds
    if impact_detected:
        for _ in range(fps * 3):  # Show for 3 seconds
            out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

logging.info(f"Analysis complete. Results saved to {csv_filename} and {output_path}")