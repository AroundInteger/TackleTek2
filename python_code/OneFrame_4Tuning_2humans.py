#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:35:43 2024
@author: Rowan
"""
import cv2
import mediapipe as mp

def process_frame(frame, min_detection_confidence, min_tracking_confidence):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                      min_tracking_confidence=min_tracking_confidence,
                      model_complexity=2) as pose:  # Use the most complex model for better accuracy
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        annotated_frame = frame.copy()
        
        if results.pose_landmarks:
            # Draw pose landmarks for the detected person
            mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # Add text to identify the person
            cv2.putText(annotated_frame, "Person", 
                        (int(results.pose_landmarks.landmark[0].x * frame.shape[1]), 
                         int(results.pose_landmarks.landmark[0].y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    return annotated_frame

def process_video(input_path, min_detection_confidence, min_tracking_confidence):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, min_detection_confidence, min_tracking_confidence)
        
        # Display the processed frame
        cv2.imshow('Pose Estimation', processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
input_video = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/R1_3.mp4"

# Adjustable parameters
min_detection_confidence = 0.6
min_tracking_confidence = 0.45

process_video(input_video, min_detection_confidence, min_tracking_confidence)