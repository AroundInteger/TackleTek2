#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:35:43 2024

@author: Rowan
"""

import cv2
import mediapipe as mp
import numpy as np

def estimate_background(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    avg_background = np.float32(frame)

    for _ in range(1, num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.accumulateWeighted(frame, avg_background, 0.1)

    background = cv2.convertScaleAbs(avg_background)
    
    cap.release()
    return background

def crop_image(image):
    height, width = image.shape[:2]
    crop_height = int(height * 2/3)
    crop_width = int(width / 2)
    
    cropped_image = image[height-crop_height:, crop_width:]
    
    return cropped_image

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8,8)):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the A and B channels
    limg = cv2.merge((cl,a,b))
    
    # Convert image back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def process_frame(frame, background, min_detection_confidence, min_tracking_confidence):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                        min_tracking_confidence=min_tracking_confidence)

    frame_diff = cv2.absdiff(frame, background)
    cropped_frame = crop_image(frame_diff)
    enhanced_frame = enhance_contrast(cropped_frame)

    frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    annotated_frame = enhanced_frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return annotated_frame, enhanced_frame

def process_video(input_path, min_detection_confidence, min_tracking_confidence):
    background = estimate_background(input_path)
    if background is None:
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    cropped_background = crop_image(background)
    processed_frame, enhanced_frame = process_frame(frame, background, min_detection_confidence, min_tracking_confidence)

    # Display both the enhanced frame and the processed frame
    cv2.imshow('Enhanced Frame', enhanced_frame)
    cv2.imshow('Pose Estimation', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

# Example usage
input_video = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/R1_2.mp4"

# Adjustable parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

process_video(input_video, min_detection_confidence, min_tracking_confidence)