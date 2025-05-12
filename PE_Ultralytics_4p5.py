#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:40:07 2024

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
model = YOLO('yolov8n-pose.pt')

# Open the video file
video_path = "R1_1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    logging.error(f"Could not open video file {video_path}")
    sys.exit(1)

# Get the video's width, height, and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

logging.info(f"Video properties: Width={width}, Height={height}, FPS={fps}")

# Create a VideoWriter object to save the output video
output_path = 'output_pose_estimation_4p5.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Check if the VideoWriter was initialized correctly
if not out.isOpened():
    logging.error(f"Could not create output video file {output_path}")
    cap.release()
    sys.exit(1)

# Prepare CSV file
csv_filename = 'pose_data_4p5.csv'
csv_header = ['frame', 'person_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'] + [f'kp_{i}_x' for i in range(17)] + [f'kp_{i}_y' for i in range(17)]

try:
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        frame_count = 0
        while True:
            # Read a frame from the video
            success, frame = cap.read()

            if not success:
                logging.info("Reached end of video or encountered an error reading frame.")
                break

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the frame into the output file
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Extract and save pose data
            for i, result in enumerate(results[0].boxes):
                bbox = result.xyxy[0].tolist()  # get bbox coordinates
                kpts = results[0].keypoints[i].xy[0].tolist()  # get keypoints
                
                # Prepare data for CSV
                person_data = [frame_count, i, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                for kp in kpts:
                    person_data.extend(kp)
                
                # Write to CSV
                csv_writer.writerow(person_data)

            frame_count += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Processing stopped by user.")
                break

except Exception as e:
    logging.exception(f"An error occurred: {str(e)}")

finally:
    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Check if the output video file was created and has content
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    logging.info(f"Annotated video saved successfully to {output_path}")
else:
    logging.error(f"The output video file {output_path} was not created or is empty")

logging.info(f"Pose data saved to {csv_filename}")