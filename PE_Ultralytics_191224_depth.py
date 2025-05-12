#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 2024

@author: rowanbrown
"""

import cv2
import numpy as np
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_orange_markers(frame):
    """
    Detect orange markers with multiple HSV ranges to handle lighting variations
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Multiple HSV ranges to handle different lighting conditions
    hsv_ranges = [
        # Bright orange
        (np.array([0, 150, 150]), np.array([20, 255, 255])),
        # Darker orange
        (np.array([0, 100, 100]), np.array([20, 255, 200])),
        # Lighter orange (sun-bleached)
        (np.array([0, 50, 200]), np.array([25, 150, 255])),
        # Reddish orange
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ]
    
    # Combined mask
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Noise reduction
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter markers
    marker_centers = []
    min_area = 100
    max_area = 10000
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
            
            if circularity > 0.6:
                marker_centers.append((int(x), int(y)))
    
    if len(marker_centers) == 4:
        # Sort by y coordinate first (top to bottom)
        marker_centers.sort(key=lambda p: p[1])
        
        # Sort top two points by x coordinate (left to right)
        top_points = sorted(marker_centers[:2], key=lambda p: p[0])
        # Sort bottom two points by x coordinate (left to right)
        bottom_points = sorted(marker_centers[2:], key=lambda p: p[0])
        
        # Combine points in order: top-left, top-right, bottom-right, bottom-left
        marker_centers = [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]
        return np.array(marker_centers)
    
    return None

def draw_marker_connections(frame, markers):
    """
    Draw markers and their connections
    """
    if markers is None or len(markers) != 4:
        return frame
    
    # Draw markers
    for i, center in enumerate(markers):
        cv2.circle(frame, tuple(center), 10, (0, 165, 255), -1)  # Solid orange circle
        cv2.circle(frame, tuple(center), 12, (255, 255, 255), 2)  # White border
        
    # Draw connections
    connections = [(0,1), (1,2), (2,3), (3,0)]  # Square connections
    for start_idx, end_idx in connections:
        cv2.line(frame, 
                 tuple(markers[start_idx]), 
                 tuple(markers[end_idx]), 
                 (255, 255, 255), 2)  # White lines
        
    return frame

def main():
    # Open the video file
    video_path = "R1_1.mp4"
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    logging.info(f"Video properties: Width={width}, Height={height}, FPS={fps}")

    # Try different codecs
    codecs = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]

    out = None
    for codec, ext in codecs:
        output_path = f'marker_detection_output{ext}'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            logging.info(f"Successfully created output video file with codec {codec}")
            break
        else:
            logging.warning(f"Failed to create output video with codec {codec}")

    if not out.isOpened():
        logging.error("Could not create output video file with any of the attempted codecs")
        cap.release()
        sys.exit(1)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Detect markers
            markers = detect_orange_markers(frame)
            
            # Draw markers and connections
            annotated_frame = draw_marker_connections(frame.copy(), markers)

            # Write and display
            out.write(annotated_frame)
            
            # Resize for display if needed
            display_frame = annotated_frame
            if width > 1920:  # If video is very wide
                scale = 1920 / width
                display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
            
            cv2.imshow("Marker Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Check if the output video file was created and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Annotated video saved successfully to {output_path}")
    else:
        logging.error(f"The output video file {output_path} was not created or is empty")

if __name__ == "__main__":
    main()