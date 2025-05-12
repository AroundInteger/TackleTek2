#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:04:52 2024

@author: rowanbrown
"""

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
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_orange_markers(frame):
    """
    #Detect orange markers with multiple HSV ranges to handle lighting variations
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv_ranges = [
        (np.array([0, 150, 150]), np.array([20, 255, 255])),
        (np.array([0, 100, 100]), np.array([20, 255, 200])),
        (np.array([0, 50, 200]), np.array([25, 150, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
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

def calibrate_markers(cap, num_frames=100):
    """
    Collect marker positions from first n frames to establish stable positions
    """
    marker_positions = defaultdict(list)  # List of positions for each vertex
    frame_count = 0
    
    logging.info("Starting marker calibration...")
    
    while frame_count < num_frames:
        success, frame = cap.read()
        if not success:
            break
            
        markers = detect_orange_markers(frame)
        if markers is not None:
            for i, pos in enumerate(markers):
                marker_positions[i].append(pos)
        
        frame_count += 1
    
    # Calculate median position for each vertex
    stable_positions = None
    if all(len(positions) > num_frames//2 for positions in marker_positions.values()):
        stable_positions = np.array([
            np.median(positions, axis=0).astype(int) 
            for positions in marker_positions.values()
        ])
        logging.info("Marker calibration completed successfully")
    else:
        logging.warning("Insufficient marker detections during calibration")
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return stable_positions

def create_grid_points(corners, grid_size=0.5):

    #Create grid points and lines with 0.5m spacing

    # Convert corners to numpy array
    corners = np.float32(corners)
    
    # Real-world coordinates (5m square)
    src = np.float32([
        [0, 0],      # Top-left
        [5, 0],      # Top-right
        [5, 5],      # Bottom-right
        [0, 5]       # Bottom-left
    ])
    
    # Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(src, corners)
    
    # Create grid points (0 to 5 meters in 0.5m steps)
    steps = np.arange(0, 5.1, grid_size)
    grid_points = []
    
    # Generate all grid intersections
    for x in steps:
        for y in steps:
            point = np.float32([x, y, 1])
            transformed = matrix.dot(point)
            grid_points.append((
                int(transformed[0]/transformed[2]), 
                int(transformed[1]/transformed[2])
            ))
    
    # Create horizontal and vertical line points
    h_lines = []  # Horizontal lines
    v_lines = []  # Vertical lines
    
    # Generate points for grid lines
    for i in steps:
        h_line = []  # Points for one horizontal line
        v_line = []  # Points for one vertical line
        
        # Create more points for smoother lines
        for t in np.linspace(0, 5, 50):
            # Horizontal line point
            h_point = np.float32([t, i, 1])
            h_transformed = matrix.dot(h_point)
            h_line.append((
                int(h_transformed[0]/h_transformed[2]),
                int(h_transformed[1]/h_transformed[2])
            ))
            
            # Vertical line point
            v_point = np.float32([i, t, 1])
            v_transformed = matrix.dot(v_point)
            v_line.append((
                int(v_transformed[0]/v_transformed[2]),
                int(v_transformed[1]/v_transformed[2])
            ))
        
        h_lines.append(h_line)
        v_lines.append(v_line)
    
    return grid_points, h_lines, v_lines

def draw_grid(frame, grid_points, h_lines, v_lines, corners):

    # Draw grid, lines, and reference markers

    # Draw grid lines
    for line in h_lines:
        # Convert line points to integer arrays
        points = np.array(line, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, (100, 100, 100), 1)
    
    for line in v_lines:
        points = np.array(line, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, (100, 100, 100), 1)
    
    # Draw grid intersection points
    for point in grid_points:
        cv2.circle(frame, point, 2, (150, 150, 150), -1)
    
    # Draw marker vertices with more prominence
    for i, corner in enumerate(corners):
        # Draw orange circle with white border
        cv2.circle(frame, tuple(corner), 10, (0, 165, 255), -1)
        cv2.circle(frame, tuple(corner), 12, (255, 255, 255), 2)
        
        # Add vertex labels
        label = f"V{i+1}"
        cv2.putText(frame, label, 
                   (corner[0] + 15, corner[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw edges between markers
    for i in range(4):
        cv2.line(frame, 
                tuple(corners[i]), 
                tuple(corners[(i+1)%4]), 
                (255, 255, 255), 2)
    
    return frame

def detect_rugby_ball(frame):
    """
    Detect rugby ball using color and shape characteristics
    Returns the center position and orientation of the ball if found
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define white/off-white range in HSV
    # Wide range to account for different lighting conditions
    lower_white = np.array([50, 50, 150])
    upper_white = np.array([180, 180, 180])
    
    # Create mask for white colors
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ball_candidates = []
    min_area = 200  # Minimum area for ball
    max_area = 3000  # Maximum area for ball
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Fit an ellipse to the contour
            if len(contour) >= 5:  # Need at least 5 points to fit ellipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                
                # Calculate aspect ratio
                aspect_ratio = max(axes) / min(axes)
                
                # Rugby balls typically have aspect ratio between 1.8 and 2.5
                if 1.8 < aspect_ratio < 2.5:
                    ball_candidates.append(ellipse)
    
    # Return the most likely ball candidate based on size and aspect ratio
    if ball_candidates:
        best_candidate = max(ball_candidates, 
                           key=lambda x: cv2.contourArea(cv2.boxPoints(x).reshape((-1,1,2))))
        return best_candidate
    
    return None

def draw_ball(frame, ball_ellipse):
    """
    Draw the detected rugby ball ellipse and its orientation
    """
    if ball_ellipse is not None:
        (center, axes, angle) = ball_ellipse
        center = tuple(map(int, center))
        axes = tuple(map(int, axes))
        
        # Draw the ellipse
        cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 0), 2)
        
        # Draw orientation line
        angle_rad = np.deg2rad(angle)
        line_length = int(max(axes))
        end_point = (
            int(center[0] + np.cos(angle_rad) * line_length),
            int(center[1] + np.sin(angle_rad) * line_length)
        )
        cv2.line(frame, center, end_point, (0, 0, 255), 2)
        
        # Add text for ball position
        cv2.putText(frame, f"Ball ({center[0]}, {center[1]})", 
                   (center[0] + 20, center[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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

    # Calibrate markers first
    stable_markers = calibrate_markers(cap)
    if stable_markers is None:
        logging.error("Marker calibration failed")
        sys.exit(1)

    # Create grid points and lines
    grid_points, h_lines, v_lines = create_grid_points(stable_markers)

    # Set up video writer
    output_path = 'marker_ball_detection.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Draw grid and markers
            annotated_frame = draw_grid(frame.copy(), grid_points, h_lines, v_lines, stable_markers)
            
            # Detect and draw ball
            ball_ellipse = detect_rugby_ball(frame)
            annotated_frame = draw_ball(annotated_frame, ball_ellipse)

            out.write(annotated_frame)
            
            # Resize for display if needed
            display_frame = annotated_frame
            if width > 1920:
                scale = 1920 / width
                display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
            
            cv2.imshow("Marker and Ball Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()