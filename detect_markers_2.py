
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
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_orange_markers(frame):
    """
    Detect orange markers with multiple HSV ranges to handle lighting variations
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv_ranges = [
        (np.array([0, 150, 150]), np.array([20, 255, 255])),  # Bright orange
        (np.array([0, 100, 100]), np.array([20, 255, 200])),  # Darker orange
        (np.array([0, 50, 200]), np.array([25, 150, 255])),   # Lighter orange
        (np.array([170, 100, 100]), np.array([180, 255, 255])) # Reddish orange
    ]
    
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
        marker_centers.sort(key=lambda p: p[1])
        top_points = sorted(marker_centers[:2], key=lambda p: p[0])
        bottom_points = sorted(marker_centers[2:], key=lambda p: p[0])
        marker_centers = [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]
        return np.array(marker_centers)
    
    return None

def calculate_grid_points(markers, grid_size=10):
    """
    Calculate grid points based on marker positions
    """
    if markers is None or len(markers) != 4:
        return None
    
    # Extract corner points
    tl, tr, br, bl = markers
    
    # Create grid points
    grid_points = []
    for i in range(grid_size + 1):
        row_points = []
        for j in range(grid_size + 1):
            # Calculate interpolation factors
            y_factor = i / grid_size
            x_factor = j / grid_size
            
            # Interpolate top and bottom points
            top_point = tl + (tr - tl) * x_factor
            bottom_point = bl + (br - bl) * x_factor
            
            # Interpolate final point
            point = top_point + (bottom_point - top_point) * y_factor
            row_points.append(point.astype(int))
        grid_points.append(row_points)
    
    return np.array(grid_points)

def draw_grid(frame, grid_points):
    """
    Draw grid lines based on grid points
    """
    if grid_points is None:
        return frame
    
    # Draw horizontal lines
    for row in grid_points:
        for i in range(len(row) - 1):
            pt1 = tuple(row[i])
            pt2 = tuple(row[i + 1])
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
    
    # Draw vertical lines
    for j in range(len(grid_points[0])):
        for i in range(len(grid_points) - 1):
            pt1 = tuple(grid_points[i][j])
            pt2 = tuple(grid_points[i + 1][j])
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
    
    return frame

def main():
    video_path = "R1_1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    logging.info(f"Video properties: Width={width}, Height={height}, FPS={fps}")

    codecs = [('mp4v', '.mp4'), ('avc1', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
    out = None
    
    for codec, ext in codecs:
        output_path = f'marker_grid_detection{ext}'
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

    # Initialize grid points storage
    initial_grids = deque(maxlen=10)
    established_grid = None

    try:
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Detect markers
            markers = detect_orange_markers(frame)
            
            # For first 10 frames, collect grid points
            if frame_count < 10:
                if markers is not None:
                    grid_points = calculate_grid_points(markers)
                    if grid_points is not None:
                        initial_grids.append(grid_points)
                frame_count += 1
                
                # After 10 frames, calculate average grid
                if frame_count == 10 and len(initial_grids) > 0:
                    established_grid = np.mean(initial_grids, axis=0).astype(int)
                    logging.info("Established reference grid from initial frames")
            
            # Draw the established grid
            if established_grid is not None:
                frame = draw_grid(frame, established_grid)
            
            # Draw current markers if detected
            if markers is not None:
                for center in markers:
                    cv2.circle(frame, tuple(center), 10, (0, 165, 255), -1)  # Orange circle
                    cv2.circle(frame, tuple(center), 12, (255, 255, 255), 2)  # White border

            # Write and display
            out.write(frame)
            
            # Resize for display if needed
            display_frame = frame
            if width > 1920:
                scale = 1920 / width
                display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            cv2.imshow("Marker Detection with Grid", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Annotated video saved successfully to {output_path}")
    else:
        logging.error(f"The output video file {output_path} was not created or is empty")

if __name__ == "__main__":
    main()