#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:26:37 2024
@author: Rowan
"""
import cv2
import mediapipe as mp
import os

def process_video(input_path, output_path, output_folder):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect poses
        results = pose.process(image_rgb)

        # Draw pose annotations on the image
        annotated_image = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write the annotated frame to the output video
        out.write(annotated_image)

        # Save the annotated frame as an image
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"), annotated_image)

        # Display the frame
        cv2.imshow('Pose Estimation', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Processed frames saved in {output_folder}")

# Example usage
input_video = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/R1_3.mp4"
output_video = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/R1_3_pose.mp4"
output_folder = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/R1_3_frames"

process_video(input_video, output_video, output_folder)



print("Finished processing video, now saved")