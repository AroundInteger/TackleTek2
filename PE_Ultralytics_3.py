#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:24:12 2024

@author: mrbimac
"""

import cv2
from ultralytics import YOLO
import numpy as np
import csv

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')
cv2.setNumThreads(0)


# Open the video file
video_path = "R1_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_pose_estimation.mp4', fourcc, 30, (width, height))

# Prepare CSV file
csv_filename = 'pose_data.csv'
csv_header = ['frame', 'person_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'] + [f'kp_{i}_x' for i in range(17)] + [f'kp_{i}_y' for i in range(17)]

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    frame_count = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
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
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Pose data saved to {csv_filename}")