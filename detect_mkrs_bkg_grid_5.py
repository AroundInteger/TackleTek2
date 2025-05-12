#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 2024
@author: rowanbrown
Modified to remove ball detection and add grid to all panels
"""

import cv2
import numpy as np
import logging
import os
import sys
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BackgroundSubtractor:
    def __init__(self, history=1000):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=50,
            detectShadows=False
        )
        self.background_model = None
        self.frames_processed = 0
        
    def update(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        self.frames_processed += 1
        
        # Update background model
        if self.frames_processed >= 100:  # Allow model to stabilize
            self.background_model = self.bg_subtractor.getBackgroundImage()
        
        return fg_mask, self.background_model
    
def enhance_frame(frame):
    # Convert to YUV colorspace
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Equalize the Y channel
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    # Convert back to BGR
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def enhance_frame_clahe(frame):
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # Apply CLAHE to L channel
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    # Convert back to BGR
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def create_binary_grid_mask(shape, grid_points, h_lines, v_lines):
    """
    Create a binary mask of the grid with thicker lines for better intersection detection
    """
    grid_mask = np.zeros(shape, dtype=np.uint8)
    
    # Draw lines into the mask with increased thickness
    for line in h_lines + v_lines:
        points = np.array(line, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(grid_mask, [points], False, 255, 2)  # Increased thickness to 2
    
    # Add grid points with larger radius
    for point in grid_points:
        x, y = map(int, point)
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            cv2.circle(grid_mask, (x, y), 3, 255, -1)  # Increased radius to 3
    
    # Dilate the mask slightly to ensure continuous lines
    kernel = np.ones((2,2), np.uint8)
    grid_mask = cv2.dilate(grid_mask, kernel, iterations=1)
    
    return grid_mask

def create_debug_view(frame, fg_mask, bg_model, grid_points, h_lines, v_lines, stable_markers):
    """
    Create a debug view showing original frame, background model, and foreground mask
    with grid overlay and intersection highlights
    """
    # Get dimensions
    height, width = frame.shape[:2]
    
    # Create binary grid mask
    grid_mask = create_binary_grid_mask(fg_mask.shape, grid_points, h_lines, v_lines)
    
    # Create intersection mask of grid and foreground
    intersection_mask = cv2.bitwise_and(grid_mask, fg_mask)
    
    # Create colored versions of masks
    mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    
    # Use original frame size for background model
    if bg_model is not None:
        bg_display = bg_model.copy()
    else:
        bg_display = np.zeros_like(frame)
    
    # Draw grid on original frame with intersection areas removed
    frame_with_grid = draw_masked_grid(frame.copy(), grid_points, h_lines, v_lines, stable_markers, fg_mask, intersection_mask)
    
    # Draw grid on background model (without removing intersections)
    bg_with_grid = draw_masked_grid(bg_display, grid_points, h_lines, v_lines, stable_markers, np.zeros_like(fg_mask))
    
    # Apply the same masked grid to the foreground view
    mask_with_grid = mask_colored.copy()
    # Extract grid by finding differences between original and grid frames
    grid_diff = cv2.absdiff(frame_with_grid, frame)
    grid_mask = np.any(grid_diff > 0, axis=2)  # Combine channels
    # Apply yellow color to grid locations
    mask_with_grid[grid_mask] = [0, 255, 255]  # Yellow color
    
    # Stack the images vertically
    debug_view = np.vstack([
        frame_with_grid,    # Original frame with grid on top
        bg_with_grid,       # Background model with grid in middle
        mask_with_grid      # Foreground mask with grid
    ])
    
    # Add labels on left side of each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_positions = [30, height + 30, 2 * height + 30]
    labels = ['Original', 'Background Model', 'Foreground Mask']
    
    for label, y_pos in zip(labels, y_positions):
        # Draw dark background for text
        (text_width, text_height), _ = cv2.getTextSize(label, font, 1.0, 2)
        cv2.rectangle(debug_view, 
                     (10, y_pos - text_height - 5),
                     (10 + text_width, y_pos + 5),
                     (0, 0, 0),
                     -1)
        # Draw text in green
        cv2.putText(debug_view, label, (10, y_pos),
                    font, 1.0, (0, 255, 0), 2)
    
    return debug_view
    
    # Stack the images vertically
    debug_view = np.vstack([
        frame_with_grid,    # Original frame with grid on top
        bg_with_grid,       # Background model with grid in middle
        mask_with_grid      # Foreground mask with grid
    ])
    
    # Add labels on left side of each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_positions = [30, height + 30, 2 * height + 30]
    labels = ['Original', 'Background Model', 'Foreground Mask']
    
    for label, y_pos in zip(labels, y_positions):
        # Draw dark background for text
        (text_width, text_height), _ = cv2.getTextSize(label, font, 1.0, 2)
        cv2.rectangle(debug_view, 
                     (10, y_pos - text_height - 5),
                     (10 + text_width, y_pos + 5),
                     (0, 0, 0),
                     -1)
        # Draw text in green
        cv2.putText(debug_view, label, (10, y_pos),
                    font, 1.0, (0, 255, 0), 2)
    
    return debug_view
    
    # Add labels on left side of each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_positions = [30, height + 30, 2 * height + 30]
    labels = ['Original', 'Background Model', 'Foreground Mask']
    
    for label, y_pos in zip(labels, y_positions):
        # Draw dark background for text
        (text_width, text_height), _ = cv2.getTextSize(label, font, 1.0, 2)
        cv2.rectangle(debug_view, 
                     (10, y_pos - text_height - 5),
                     (10 + text_width, y_pos + 5),
                     (0, 0, 0),
                     -1)
        # Draw text in green
        cv2.putText(debug_view, label, (10, y_pos),
                    font, 1.0, (0, 255, 0), 2)
    
    return debug_view
    
    # Stack the images vertically
    debug_view = np.vstack([
        frame_with_grid,    # Original frame with grid on top
        bg_with_grid,       # Background model with grid in middle
        mask_with_grid      # Foreground mask with grid on bottom
    ])
    
    # Add labels on left side of each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_positions = [30, height + 30, 2 * height + 30]
    labels = ['Original', 'Background Model', 'Foreground Mask']
    
    for label, y_pos in zip(labels, y_positions):
        # Draw dark background for text
        (text_width, text_height), _ = cv2.getTextSize(label, font, 1.0, 2)
        cv2.rectangle(debug_view, 
                     (10, y_pos - text_height - 5),
                     (10 + text_width, y_pos + 5),
                     (0, 0, 0),
                     -1)
        # Draw text in green
        cv2.putText(debug_view, label, (10, y_pos),
                    font, 1.0, (0, 255, 0), 2)
    
    return debug_view

def detect_orange_markers(frame):
    """
    Detect orange markers with multiple HSV ranges to handle lighting variations
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
    marker_positions = defaultdict(list)
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
    
    stable_positions = None
    if all(len(positions) > num_frames//2 for positions in marker_positions.values()):
        stable_positions = np.array([
            np.median(positions, axis=0).astype(int) 
            for positions in marker_positions.values()
        ])
        logging.info("Marker calibration completed successfully")
    else:
        logging.warning("Insufficient marker detections during calibration")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return stable_positions

def create_grid_points(corners, grid_size=0.5):
    """
    Create grid points and lines with 0.5m spacing
    """
    corners = np.float32(corners)
    
    src = np.float32([
        [0, 0],
        [5, 0],
        [5, 5],
        [0, 5]
    ])
    
    matrix = cv2.getPerspectiveTransform(src, corners)
    
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
    
    h_lines = []
    v_lines = []
    
    for i in steps:
        h_line = []
        v_line = []
        
        for t in np.linspace(0, 5, 50):
            h_point = np.float32([t, i, 1])
            h_transformed = matrix.dot(h_point)
            h_line.append((
                int(h_transformed[0]/h_transformed[2]),
                int(h_transformed[1]/h_transformed[2])
            ))
            
            v_point = np.float32([i, t, 1])
            v_transformed = matrix.dot(v_point)
            v_line.append((
                int(v_transformed[0]/v_transformed[2]),
                int(v_transformed[1]/v_transformed[2])
            ))
        
        h_lines.append(h_line)
        v_lines.append(v_line)
    
    return grid_points, h_lines, v_lines

def draw_masked_grid(frame, grid_points, h_lines, v_lines, corners, fg_mask, intersection_mask=None):
    """
    Draw grid only in background areas by using the foreground mask and intersection mask
    """
    # Create a clean copy of the frame for grid drawing
    grid_frame = frame.copy()
    
    # Create binary grid mask for the full grid
    grid_mask = create_binary_grid_mask(fg_mask.shape, grid_points, h_lines, v_lines)
    
    # Create mask for edges
    edge_mask = np.zeros_like(fg_mask)
    for i in range(4):
        cv2.line(edge_mask, 
                tuple(corners[i]), 
                tuple(corners[(i+1)%4]), 
                255, 2)
    
    # Combine grid and edge masks
    combined_mask = cv2.bitwise_or(grid_mask, edge_mask)
    
    # If intersection_mask is provided, remove those areas from the grid
    if intersection_mask is not None:
        # Dilate intersection mask slightly to create a buffer
        kernel = np.ones((3,3), np.uint8)
        intersection_dilated = cv2.dilate(intersection_mask, kernel, iterations=1)
        combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(intersection_dilated))
    
    # Dilate the foreground mask to create a buffer around moving objects
    kernel = np.ones((3,3), np.uint8)
    fg_mask_dilated = cv2.dilate(fg_mask, kernel, iterations=2)
    
    # Invert the dilated foreground mask to get background mask
    bg_mask = cv2.bitwise_not(fg_mask_dilated)
    
    # Combine masks
    valid_mask = cv2.bitwise_and(combined_mask, bg_mask)
    
    # Draw the valid grid and edges onto the frame
    grid_frame[valid_mask > 0] = [100, 100, 100]
    
    # Draw marker vertices over everything
    for i, corner in enumerate(corners):
        cv2.circle(grid_frame, tuple(corner), 10, (0, 165, 255), -1)  # Orange center
        cv2.putText(grid_frame, f"V{i+1}", 
                   (corner[0] + 15, corner[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    return grid_frame
    
    # Draw marker vertices and their connections
    for i, corner in enumerate(corners):
        cv2.circle(grid_frame, tuple(corner), 10, (0, 165, 255), -1)
        cv2.circle(grid_frame, tuple(corner), 12, (255, 255, 255), 2)
        
        label = f"V{i+1}"
        cv2.putText(grid_frame, label, 
                   (corner[0] + 15, corner[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw edges between markers
    for i in range(4):
        cv2.line(grid_frame, 
                tuple(corners[i]), 
                tuple(corners[(i+1)%4]), 
                (255, 255, 255), 2)
    
    return grid_frame

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

    # Calibrate markers
    stable_markers = calibrate_markers(cap)
    if stable_markers is None:
        logging.error("Calibration failed")
        sys.exit(1)

    # Create grid points and lines
    grid_points, h_lines, v_lines = create_grid_points(stable_markers)

    # Initialize background subtractor
    bg_subtractor = BackgroundSubtractor()

    # Set up video writer for debug view (3x height of original frame)
    output_path = 'rugby_analysis_output.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height * 3))

    # Create named window
    cv2.namedWindow('Rugby Analysis', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            success, frame = cap.read()

            # enhance video
            frame1 = enhance_frame(frame)
            frame2 = enhance_frame_clahe(frame1)
            if not success:
                break

            # Get foreground mask and background model
            fg_mask, bg_model = bg_subtractor.update(frame)
            
            # Create debug visualization with grid overlay on all panels
            debug_view = create_debug_view(frame, fg_mask, bg_model, 
                                         grid_points, h_lines, v_lines, stable_markers)
            
            # Write debug view to output video
            out.write(debug_view)
            
            # Resize for display if needed
            if debug_view.shape[1] > 1920:
                scale = 1920 / debug_view.shape[1]
                display_frame = cv2.resize(debug_view, None, fx=scale, fy=scale)
            else:
                display_frame = debug_view
            
            # Show frames
            cv2.imshow('Debug View', debug_view)
            cv2.imshow('Rugby Analysis', display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")

    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Check if output video was created successfully
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Analysis video saved successfully to {output_path}")
    else:
        logging.error(f"The output video file {output_path} was not created or is empty")

if __name__ == "__main__":
    main()