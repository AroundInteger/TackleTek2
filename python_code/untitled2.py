#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:06:11 2024

@author: Rowan
"""

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect poses
    results = pose.process(image_rgb)

    # Draw pose annotations on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image
    cv2.imshow('MediaPipe Pose', image)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()