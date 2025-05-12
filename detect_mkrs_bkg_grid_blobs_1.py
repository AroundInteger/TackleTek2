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
from collections import defaultdict, deque
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize YOLO model
model = YOLO('yolov8x-pose.pt')  # Using the most accurate model

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
        self.model = YOLO('yolov8x-pose.pt')  # Using the most accurate model
        self.keypoints = None
        self.bboxes = None
        self.prev_ball_carrier = None
        self.prev_tackler = None
        
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
        
        logging.info(f"Detected {len(self.keypoints)} poses in the frame.")
        return self.keypoints, self.bboxes
    
    def identify_players(self, keypoints):
        poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
        
        if len(poses) == 2:
            # If we have two poses, identify based on hip position
            left_player = min(poses, key=lambda x: x[11, 0])  # Left hip x-coordinate
            right_player = max(poses, key=lambda x: x[11, 0])  # Left hip x-coordinate
            return left_player, right_player
        elif len(poses) == 1:
            # If we have only one pose, assign it based on minimum distance to previous positions
            pose = poses[0]
            dist_to_ball_carrier = np.linalg.norm(pose[11] - self.prev_ball_carrier[11])
            dist_to_tackler = np.linalg.norm(pose[11] - self.prev_tackler[11])
            
            if dist_to_ball_carrier < dist_to_tackler:
                return pose, self.prev_tackler
            else:
                return self.prev_ball_carrier, pose
        else:
            # If we don't have any poses, return the previous positions
            return self.prev_ball_carrier, self.prev_tackler
    
    def draw_poses(self, frame):
        """Draw detected poses on the frame"""
        if self.keypoints is None or self.bboxes is None:
            return frame
        
        output_frame = frame.copy()
        
        # Identify players
        ball_carrier, tackler = self.identify_players(self.keypoints)
        
        if ball_carrier is not None and tackler is not None:
            self.prev_ball_carrier, self.prev_tackler = ball_carrier, tackler
            
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

def draw_grid(frame, grid_width, grid_height, cell_size):
    """Draw a grid on the frame."""
    for x in range(0, grid_width, cell_size):
        cv2.line(frame, (x, 0), (x, grid_height), (100, 100, 100), 1)
    for y in range(0, grid_height, cell_size):
        cv2.line(frame, (0, y), (grid_width, y), (100, 100, 100), 1)

def initialize_players(keypoints):
    poses = [kp.xy[0].cpu().numpy() for kp in keypoints]
    if len(poses) >= 2:
        left_player = min(poses, key=lambda x: x[11, 0])
        right_player = max(poses, key=lambda x: x[11, 0])
        return left_player, right_player
    else:
        return None, None

def identify_players(keypoints, prev_ball_carrier, prev_tackler):
    """Identify which detected pose corresponds to the ball carrier and tackler."""
    if keypoints is None or len(keypoints) == 0:
        return prev_ball_carrier, prev_tackler
        
    poses = keypoints.data.cpu().numpy()
    if len(poses) == 0:
        return prev_ball_carrier, prev_tackler
        
    # If we have previous positions, use them to identify players
    if prev_ball_carrier is not None and prev_tackler is not None:
        # Calculate distances to previous positions (use only x, y)
        dists_to_ball_carrier = [np.linalg.norm(pose[11][:2] - prev_ball_carrier[11][:2]) for pose in poses]
        dists_to_tackler = [np.linalg.norm(pose[11][:2] - prev_tackler[11][:2]) for pose in poses]
        
        # Find closest poses to previous positions
        ball_carrier_idx = np.argmin(dists_to_ball_carrier)
        tackler_idx = np.argmin(dists_to_tackler)
        
        # If same pose is closest to both, use second closest for one
        if ball_carrier_idx == tackler_idx:
            if len(poses) > 1:
                # Use second closest for tackler
                tackler_dists = dists_to_tackler.copy()
                tackler_dists[ball_carrier_idx] = float('inf')
                tackler_idx = np.argmin(tackler_dists)
            else:
                # If only one pose, use it for both
                return poses[0], poses[0]
        
        return poses[ball_carrier_idx], poses[tackler_idx]
    
    # If no previous positions, use leftmost and rightmost poses
    left_hip_x = [pose[11][0] for pose in poses]
    ball_carrier_idx = np.argmin(left_hip_x)  # Leftmost is ball carrier
    tackler_idx = np.argmax(left_hip_x)  # Rightmost is tackler
    
    return poses[ball_carrier_idx], poses[tackler_idx]

def draw_annotations(frame, ball_carrier, tackler, ball_carrier_history, tackler_history):
    colors = {
        'ball_carrier_wrist': (0, 255, 0),
        'ball_carrier_hip': (255, 0, 255),
        'ball_carrier_knee': (255, 255, 0),
        'tackler_head': (255, 0, 0),
        'tackler_shoulder': (0, 165, 255),
        'tackler_hip': (128, 0, 128)
    }
    for pos in ball_carrier_history:
        cv2.circle(frame, (int(pos[9][0]), int(pos[9][1])), 3, colors['ball_carrier_wrist'], -1)
        cv2.circle(frame, (int(pos[10][0]), int(pos[10][1])), 3, colors['ball_carrier_wrist'], -1)
        cv2.circle(frame, (int(pos[11][0]), int(pos[11][1])), 3, colors['ball_carrier_hip'], -1)
        cv2.circle(frame, (int(pos[12][0]), int(pos[12][1])), 3, colors['ball_carrier_hip'], -1)
        cv2.circle(frame, (int(pos[13][0]), int(pos[13][1])), 3, colors['ball_carrier_knee'], -1)
        cv2.circle(frame, (int(pos[14][0]), int(pos[14][1])), 3, colors['ball_carrier_knee'], -1)
    for pos in tackler_history:
        cv2.circle(frame, (int(pos[0][0]), int(pos[0][1])), 3, colors['tackler_head'], -1)
        cv2.circle(frame, (int(pos[5][0]), int(pos[5][1])), 3, colors['tackler_shoulder'], -1)
        cv2.circle(frame, (int(pos[6][0]), int(pos[6][1])), 3, colors['tackler_shoulder'], -1)
        cv2.circle(frame, (int(pos[11][0]), int(pos[11][1])), 3, colors['tackler_hip'], -1)
        cv2.circle(frame, (int(pos[12][0]), int(pos[12][1])), 3, colors['tackler_hip'], -1)
    cv2.circle(frame, (int(ball_carrier[9][0]), int(ball_carrier[9][1])), 8, colors['ball_carrier_wrist'], -1)
    cv2.circle(frame, (int(ball_carrier[10][0]), int(ball_carrier[10][1])), 8, colors['ball_carrier_wrist'], -1)
    cv2.circle(frame, (int(tackler[0][0]), int(tackler[0][1])), 8, colors['tackler_head'], -1)
    wrist_height = np.median([ball_carrier[9][1], ball_carrier[10][1]])
    shoulder_height = np.median([tackler[5][1], tackler[6][1]])
    height_difference = shoulder_height - wrist_height
    diff_color = (0, 255, 0) if height_difference < 0 else (0, 0, 255)
    cv2.putText(frame, f"Shoulder-Wrist Diff: {height_difference:.2f}", (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_color, 2)
    return height_difference, shoulder_height, np.median([ball_carrier[11][1], ball_carrier[12][1]])

def calibrate_markers(video_path, num_frames=10):
    """Calibrate markers using the first few frames of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logging.info(f"Video properties: Width={width}, Height={height}, FPS={fps}")
    
    # Initialize lists to store marker positions
    marker_positions = []
    frame_count = 0
    
    # Initialize pose detection variables
    prev_ball_carrier = None
    prev_tackler = None
    ball_carrier_history = deque(maxlen=30)
    tackler_history = deque(maxlen=30)
    
    # Store the last processed frame for visualization
    last_frame = None
    
    logging.info("Starting marker calibration...")
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame for markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and store marker positions
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Adjust threshold as needed
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    marker_positions.append((cx, cy))
        
        # Process frame for pose detection
        results = model(frame)
        if len(results[0].keypoints) >= 1:
            if prev_ball_carrier is None or prev_tackler is None:
                # Initialize players for the first frame with detections
                ball_carrier, tackler = initialize_players(results[0].keypoints)
                if ball_carrier is not None and tackler is not None:
                    prev_ball_carrier, prev_tackler = ball_carrier, tackler
                    ball_carrier_history.append(ball_carrier)
                    tackler_history.append(tackler)
            else:
                # Use simplified identification for subsequent frames
                ball_carrier, tackler = identify_players(results[0].keypoints, prev_ball_carrier, prev_tackler)
                if ball_carrier is not None and tackler is not None:
                    prev_ball_carrier, prev_tackler = ball_carrier, tackler
                    ball_carrier_history.append(ball_carrier)
                    tackler_history.append(tackler)
        
        # Store the last frame for visualization
        last_frame = frame.copy()
        frame_count += 1
    
    cap.release()
    
    if not marker_positions:
        raise ValueError("No markers detected during calibration")
    
    # Calculate average marker positions
    avg_marker_positions = np.mean(marker_positions, axis=0)
    
    # Calculate grid parameters
    grid_width = width
    grid_height = height
    cell_size = 50  # Adjust as needed
    
    # Create visualization of the last frame with grid and poses
    if last_frame is not None:
        # Draw grid
        draw_grid(last_frame, grid_width, grid_height, cell_size)
        
        # Draw pose annotations if we have detected poses
        if prev_ball_carrier is not None and prev_tackler is not None:
            height_difference, shoulder_height, waist_height = draw_annotations(
                last_frame, prev_ball_carrier, prev_tackler, 
                ball_carrier_history, tackler_history)
        
        # Save the visualization
        cv2.imwrite('calibration_visualization.jpg', last_frame)
        logging.info("Saved calibration visualization to calibration_visualization.jpg")
    
    logging.info("Marker calibration completed successfully")
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'cell_size': cell_size,
        'marker_positions': marker_positions,
        'ball_carrier_history': ball_carrier_history,
        'tackler_history': tackler_history,
        'prev_ball_carrier': prev_ball_carrier,
        'prev_tackler': prev_tackler
    }

def process_video(video_path, calibration_data):
    """Process the video with the calibrated parameters."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties from calibration
    width = calibration_data['width']
    height = calibration_data['height']
    fps = calibration_data['fps']
    grid_width = calibration_data['grid_width']
    grid_height = calibration_data['grid_height']
    cell_size = calibration_data['cell_size']
    
    # Initialize pose tracking variables
    ball_carrier_history = calibration_data['ball_carrier_history']
    tackler_history = calibration_data['tackler_history']
    prev_ball_carrier = calibration_data['prev_ball_carrier']
    prev_tackler = calibration_data['prev_tackler']
    
    # Prepare output video
    output_path = 'rugby_analysis_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Process frame for pose detection
        results = model(frame)
        if len(results[0].keypoints) >= 1:
            ball_carrier, tackler = identify_players(results[0].keypoints, prev_ball_carrier, prev_tackler)
            if ball_carrier is not None and tackler is not None:
                prev_ball_carrier, prev_tackler = ball_carrier, tackler
                ball_carrier_history.append(ball_carrier)
                tackler_history.append(tackler)
                
                # Draw pose annotations
                height_difference, shoulder_height, waist_height = draw_annotations(
                    frame, ball_carrier, tackler, ball_carrier_history, tackler_history)
        
        # Draw grid
        draw_grid(frame, grid_width, grid_height, cell_size)
        
        # Write frame to output video
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    logging.info("Analysis video saved successfully to %s", output_path)

def main():
    video_path = "R1_1.mp4"
    
    try:
        # Calibrate markers and initialize pose detection
        calibration_data = calibrate_markers(video_path)
        
        # Process the video with calibrated parameters
        process_video(video_path, calibration_data)
        
    except Exception as e:
        logging.error("Error processing video: %s", str(e))
        raise

if __name__ == "__main__":
    main()

    #/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research