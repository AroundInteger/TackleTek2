#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:48:09 2024

@author: mrbimac
"""

import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')  # Using the most accurate model

def calculate_speed(prev_pos, curr_pos, fps):
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) * fps

def detect_impact(player1, player2, threshold=50):
    return np.linalg.norm(np.array(player1) - np.array(player2)) < threshold

def analyze_tackle(ball_carrier, tackler):
    # Check if tackler's head is below ball carrier's wrists
    head_below_wrists = tackler[0][1] > ball_carrier[9][1] and tackler[0][1] > ball_carrier[10][1]
    
    # Check if tackler's head, back, and hips are inline
    head_back_hips_inline = abs(tackler[0][0] - tackler[5][0]) < 20 and abs(tackler[0][0] - tackler[11][0]) < 20
    
    return head_below_wrists, head_back_hips_inline

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
csv_filename = 'tackle_analysis.csv'
csv_header = ['frame', 'ball_carrier_speed', 'tackler_speed', 'impact_detected', 'head_below_wrists', 'head_back_hips_inline']

prev_ball_carrier_pos = None
prev_tackler_pos = None
impact_detected = False
t_impact = None

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        if len(results[0].keypoints) == 2:  # Ensure we have detected two people
            ball_carrier = results[0].keypoints[0].xy[0].cpu().numpy()
            tackler = results[0].keypoints[1].xy[0].cpu().numpy()

            # Calculate speeds
            ball_carrier_speed = 0
            tackler_speed = 0
            if prev_ball_carrier_pos is not None:
                ball_carrier_speed = calculate_speed(prev_ball_carrier_pos, ball_carrier[0], fps)
                tackler_speed = calculate_speed(prev_tackler_pos, tackler[0], fps)

            # Detect impact
            if not impact_detected:
                impact_detected = detect_impact(ball_carrier[0], tackler[0])
                if impact_detected:
                    t_impact = frame_count / fps

            # Analyze tackle
            head_below_wrists, head_back_hips_inline = analyze_tackle(ball_carrier, tackler)

            # Write data to CSV
            csv_writer.writerow([frame_count, ball_carrier_speed, tackler_speed, impact_detected, head_below_wrists, head_back_hips_inline])

            # Update previous positions
            prev_ball_carrier_pos = ball_carrier[0]
            prev_tackler_pos = tackler[0]

            # Annotate frame with analysis results
            cv2.putText(annotated_frame, f"Ball Carrier Speed: {ball_carrier_speed:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Tackler Speed: {tackler_speed:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if impact_detected:
                cv2.putText(annotated_frame, f"Impact Detected at t={t_impact:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Head Below Wrists: {head_below_wrists}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Head-Back-Hips Inline: {head_back_hips_inline}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(annotated_frame)
        cv2.imshow("Rugby Tackle Analysis", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

logging.info(f"Analysis complete. Results saved to {csv_filename} and {output_path}")