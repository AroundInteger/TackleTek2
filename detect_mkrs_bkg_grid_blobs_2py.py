#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 2024
@author: rowanbrown
Modified to remove ball detection and add grid to all panels
Added YOLOv8 pose detection
"""

import cv2
import numpy as np
import logging
import os
import sys
from collections import defaultdict
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RugbyBackgroundSubtractor:
    def __init__(self, history=500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=60,
            detectShadows=False
        )
        self.prev_centers = []
        self.background_model = None
        self.frames_processed = 0
        
        # Define size constraints for rugby players
        self.min_area = 3000  # Adjust based on your video
        self.max_area = 500000  # Adjust based on your video
        
        # Morphological kernel for mask cleanup
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        
    def filter_by_size(self, mask):
        """Filter blobs by expected person size"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
    def find_two_largest_blobs(self, mask):
        """Extract exactly two largest moving objects"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Take two largest contours
        refined_mask = np.zeros_like(mask)
        if len(contours) >= 2:
            cv2.drawContours(refined_mask, contours[:2], -1, 255, -1)
            
        return refined_mask
    
    def track_players(self, curr_mask):
        """Track two players moving towards each other"""
        contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curr_centers = []
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                curr_centers.append((cx, cy))
        
        # Sort centers left to right
        curr_centers = sorted(curr_centers, key=lambda x: x[0])
        
        # Check if players are moving towards each other
        valid_movement = False
        if len(self.prev_centers) == 2 and len(curr_centers) == 2:
            left_movement = curr_centers[0][0] - self.prev_centers[0][0]  # Should be positive
            right_movement = curr_centers[1][0] - self.prev_centers[1][0]  # Should be negative
            
            if left_movement > 0 and right_movement < 0:
                valid_movement = True
        
        self.prev_centers = curr_centers
        return valid_movement
    
    def update(self, frame):
        """Update background model and get foreground mask"""
        # Initial mask from GMM
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Basic morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Filter by size
        filtered_mask = self.filter_by_size(fg_mask)
        
        # Ensure exactly two blobs
        two_player_mask = self.find_two_largest_blobs(filtered_mask)
        
        # Track movement
        valid_movement = self.track_players(two_player_mask)
        
        # Update frame counter
        self.frames_processed += 1
        
        # Update background model
        if self.frames_processed >= 100:
            self.background_model = self.bg_subtractor.getBackgroundImage()
        
        # If valid movement pattern detected, use two_player_mask
        if valid_movement:
            return two_player_mask, self.background_model
        else:
            # Fall back to filtered mask
            return filtered_mask, self.background_model

class PoseDetector:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        self.keypoints = None
        self.bboxes = None
        
    def detect(self, frame):
        """Detect human poses in the frame"""
        results = self.model(frame, verbose=False)
        
        # Extract keypoints and bounding boxes
        self.keypoints = []
        self.bboxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # Class 0 is person in COCO dataset
                    self.bboxes.append(box.xyxy[0].cpu().numpy())
                    if hasattr(box, 'keypoints'):
                        self.keypoints.append(box.keypoints[0].cpu().numpy())
        
        return self.keypoints, self.bboxes
    
    def draw_poses(self, frame):
        """Draw detected poses on the frame"""
        if self.keypoints is None or self.bboxes is None:
            return frame
        
        output_frame = frame.copy()
        
        # Draw keypoints and connections
        for keypoints, bbox in zip(self.keypoints, self.bboxes):
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            for kp in keypoints:
                x, y = map(int, kp[:2])
                cv2.circle(output_frame, (x, y), 4, (0, 0, 255), -1)
            
            # Draw skeleton connections
            skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
            
            for connection in skeleton:
                start_idx, end_idx = connection
                if keypoints[start_idx-1][2] > 0.5 and keypoints[end_idx-1][2] > 0.5:
                    start_point = tuple(map(int, keypoints[start_idx-1][:2]))
                    end_point = tuple(map(int, keypoints[end_idx-1][:2]))
                    cv2.line(output_frame, start_point, end_point, (255, 0, 0), 2)
        
        return output_frame

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

def create_debug_view(frame, fg_mask, bg_model, grid_points, h_lines, v_lines, stable_markers, pose_detector):
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
    
    # Add pose detection to the frame
    frame_with_poses = pose_detector.draw_poses(frame_with_grid)
    
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
        frame_with_poses,    # Original frame with poses and grid
        bg_with_grid,       # Background model with grid
        mask_with_grid      # Foreground mask with grid
    ])
    
    # Add labels on left side of each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_positions = [30, height + 30, 2 * height + 30]
    labels = ['Original with Poses', 'Background Model', 'Foreground Mask']
    
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
    video_path = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/R1_1.mp4"
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

    # Initialize enhanced background subtractor
    bg_subtractor = RugbyBackgroundSubtractor()

    # Initialize pose detector
    pose_detector = PoseDetector()

    # Set up video writer for debug view (3x height of original frame)
    output_path = 'rugby_analysis_output.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height * 3))

    # Create named window
    cv2.namedWindow('Rugby Analysis', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Get foreground mask and background model
            fg_mask, bg_model = bg_subtractor.update(frame)
            
            # Detect poses
            pose_detector.detect(frame)
            
            # Create debug visualization with grid overlay and poses
            debug_view = create_debug_view(frame, fg_mask, bg_model, 
                                         grid_points, h_lines, v_lines, 
                                         stable_markers, pose_detector)
            
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

    #/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research