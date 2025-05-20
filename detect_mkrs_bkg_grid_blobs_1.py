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

def calculate_perspective_transform(corners):
    """
    Calculate the perspective transform matrix from pixel coordinates to real-world coordinates.
    corners should be in order: [0,0], [5,0], [5,5], [0,5] in real-world coordinates.
    Returns the transform matrix and its inverse.
    """
    # Ensure corners are in the correct format (4x2 float32 array)
    corners = np.float32(corners).reshape(-1, 2)
    
    # Real-world coordinates (in meters)
    real_world_points = np.float32([
        [0, 0],    # Bottom left [0,0]
        [5, 0],    # Bottom right [5,0]
        [5, 5],    # Top right [5,5]
        [0, 5]     # Top left [0,5]
    ])
    
    # Calculate perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(corners, real_world_points)
    inverse_matrix = cv2.getPerspectiveTransform(real_world_points, corners)
    
    return transform_matrix, inverse_matrix

def pixel_to_real_world(pixel_coords, transform_matrix):
    """
    Convert pixel coordinates to real-world coordinates (x,y,z).
    pixel_coords: (x,y) tuple or numpy array of pixel coordinates
    transform_matrix: perspective transform matrix
    Returns: (x,y,z) tuple of real-world coordinates in meters
    """
    # Convert to numpy array if tuple
    if isinstance(pixel_coords, tuple):
        pixel_coords = np.array([pixel_coords], dtype=np.float32)
    
    # Add homogeneous coordinate
    pixel_coords_h = np.concatenate([pixel_coords, np.ones((len(pixel_coords), 1))], axis=1)
    
    # Apply transform
    real_world_coords = np.dot(pixel_coords_h, transform_matrix.T)
    
    # Normalize homogeneous coordinates
    real_world_coords = real_world_coords[:, :2] / real_world_coords[:, 2:]
    
    # Calculate z-coordinate (depth) based on y-coordinate
    # This assumes the camera is positioned at the midpoint of the 5m area
    z_coords = 2.5 - real_world_coords[:, 1]  # 2.5m is the midpoint
    
    # Combine x,y,z coordinates
    result = np.column_stack([real_world_coords, z_coords])
    
    return result[0] if len(result) == 1 else result

def real_world_to_pixel(real_world_coords, inverse_matrix):
    """
    Convert real-world coordinates (x,y,z) to pixel coordinates.
    real_world_coords: (x,y,z) tuple or numpy array of real-world coordinates in meters
    inverse_matrix: inverse perspective transform matrix
    Returns: (x,y) tuple of pixel coordinates
    """
    # Convert to numpy array if tuple
    if isinstance(real_world_coords, tuple):
        real_world_coords = np.array([real_world_coords], dtype=np.float32)
    
    # Use only x,y coordinates for the transform
    xy_coords = real_world_coords[:, :2]
    
    # Add homogeneous coordinate
    xy_coords_h = np.concatenate([xy_coords, np.ones((len(xy_coords), 1))], axis=1)
    
    # Apply inverse transform
    pixel_coords = np.dot(xy_coords_h, inverse_matrix.T)
    
    # Normalize homogeneous coordinates
    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:]
    
    return pixel_coords[0] if len(pixel_coords) == 1 else pixel_coords

def create_grid_points(corners, grid_size=0.5):
    """
    Create grid points and lines for a 5x5m grid with camera positioned midway.
    corners should be in order: [0,0], [5,0], [5,5], [0,5] in real-world coordinates.
    """
    # Calculate perspective transform matrices
    transform_matrix, inverse_matrix = calculate_perspective_transform(corners)
    
    # Ensure corners are in the correct order
    corners = np.float32(corners)
    bottom_left = corners[0]    # [0,0]
    bottom_right = corners[1]   # [5,0]
    top_right = corners[2]      # [5,5]
    top_left = corners[3]       # [0,5]
    
    # Number of grid lines for 0.5m spacing (including edges)
    num_lines_coarse = 11  # 0, 0.5, 1, ..., 5
    
    # Number of grid lines for 0.25m spacing (including edges)
    num_lines_fine = 21  # 0, 0.25, 0.5, ..., 5
    
    # Create vertical lines (parallel to y-axis)
    v_lines_coarse = []
    v_lines_fine = []
    
    # Create coarse lines (0.5m spacing)
    for i in range(num_lines_coarse):
        t = i / (num_lines_coarse - 1)
        bottom_point = bottom_left + t * (bottom_right - bottom_left)
        top_point = top_left + t * (top_right - top_left)
        
        v_line = []
        for s in np.linspace(0, 1, 50):
            point = bottom_point + s * (top_point - bottom_point)
            v_line.append((int(point[0]), int(point[1])))
        v_lines_coarse.append(v_line)
    
    # Create fine lines (0.25m spacing)
    for i in range(num_lines_fine):
        t = i / (num_lines_fine - 1)
        bottom_point = bottom_left + t * (bottom_right - bottom_left)
        top_point = top_left + t * (top_right - top_left)
        
        v_line = []
        for s in np.linspace(0, 1, 50):
            point = bottom_point + s * (top_point - bottom_point)
            v_line.append((int(point[0]), int(point[1])))
        v_lines_fine.append(v_line)
    
    # Create horizontal lines (parallel to x-axis)
    h_lines_coarse = []
    h_lines_fine = []
    
    # Create coarse lines (0.5m spacing)
    for i in range(num_lines_coarse):
        t = i / (num_lines_coarse - 1)
        left_point = bottom_left + t * (top_left - bottom_left)
        right_point = bottom_right + t * (top_right - bottom_right)
        
        h_line = []
        for s in np.linspace(0, 1, 50):
            point = left_point + s * (right_point - left_point)
            h_line.append((int(point[0]), int(point[1])))
        h_lines_coarse.append(h_line)
    
    # Create fine lines (0.25m spacing)
    for i in range(num_lines_fine):
        t = i / (num_lines_fine - 1)
        left_point = bottom_left + t * (top_left - bottom_left)
        right_point = bottom_right + t * (top_right - bottom_right)
        
        h_line = []
        for s in np.linspace(0, 1, 50):
            point = left_point + s * (right_point - left_point)
            h_line.append((int(point[0]), int(point[1])))
        h_lines_fine.append(h_line)
    
    # Create grid points at intersections of coarse lines
    grid_points = []
    for h_line, v_line in zip(h_lines_coarse, v_lines_coarse):
        for point in h_line:
            grid_points.append(point)
    
    return (grid_points, h_lines_coarse, v_lines_coarse, h_lines_fine, v_lines_fine, 
            transform_matrix, inverse_matrix)

def create_ground_plane_mask(frame, keypoints):
    """Create a mask for the ground plane based on player positions."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for pose in keypoints:
        # Get ankle positions (keypoints 15 and 16)
        if len(pose) > 16:  # Ensure we have enough keypoints
            left_ankle = pose[15][:2]  # Left ankle
            right_ankle = pose[16][:2]  # Right ankle
            
            # Create polygon points
            points = np.array([
                left_ankle,
                right_ankle,
                [right_ankle[0], frame.shape[0]],  # Bottom right
                [left_ankle[0], frame.shape[0]]    # Bottom left
            ], dtype=np.int32)
            
            # Fill the polygon
            cv2.fillPoly(mask, [points], 255)
    
    return mask

def draw_masked_grid(frame, grid_points, h_lines_coarse, v_lines_coarse, 
                    h_lines_fine, v_lines_fine, stable_markers, fg_mask, 
                    intersection_mask, grid_size=0.5):
    """Draw grid with proper perspective and masking."""
    # Create a copy of the frame
    output = frame.copy()
    
    # Create ground plane mask from pose keypoints
    ground_plane_mask = create_ground_plane_mask(frame, stable_markers)
    
    # Create separate masks for fine and coarse grids
    fine_mask = np.zeros_like(ground_plane_mask)
    coarse_mask = np.zeros_like(ground_plane_mask)
    
    # Draw fine grid lines first (0.25m spacing)
    # Use consistent parameters for both horizontal and vertical lines
    fine_line_params = {
        'thickness': 1,
        'intensity': 100,  # Lighter color for fine lines
        'depth_factor': 0.3
    }
    
    # Draw horizontal fine lines
    for line in h_lines_fine:
        depth_factor = line[0][1] / frame.shape[0]
        intensity = int(fine_line_params['intensity'] * (1 - depth_factor * fine_line_params['depth_factor']))
        
        for i in range(len(line) - 1):
            pt1 = tuple(map(int, line[i]))
            pt2 = tuple(map(int, line[i + 1]))
            cv2.line(fine_mask, pt1, pt2, 255, fine_line_params['thickness'])
    
    # Draw vertical fine lines with same parameters
    for line in v_lines_fine:
        depth_factor = line[0][1] / frame.shape[0]
        intensity = int(fine_line_params['intensity'] * (1 - depth_factor * fine_line_params['depth_factor']))
        
        for i in range(len(line) - 1):
            pt1 = tuple(map(int, line[i]))
            pt2 = tuple(map(int, line[i + 1]))
            cv2.line(fine_mask, pt1, pt2, 255, fine_line_params['thickness'])
    
    # Draw coarse grid lines (0.5m spacing)
    # Use consistent parameters for both horizontal and vertical lines
    coarse_line_params = {
        'thickness': 2,
        'intensity': 255,  # Brighter color for coarse lines
        'depth_factor': 0.3
    }
    
    # Draw horizontal coarse lines
    for line in h_lines_coarse:
        depth_factor = line[0][1] / frame.shape[0]
        intensity = int(coarse_line_params['intensity'] * (1 - depth_factor * coarse_line_params['depth_factor']))
        
        for i in range(len(line) - 1):
            pt1 = tuple(map(int, line[i]))
            pt2 = tuple(map(int, line[i + 1]))
            cv2.line(coarse_mask, pt1, pt2, 255, coarse_line_params['thickness'])
    
    # Draw vertical coarse lines with same parameters
    for line in v_lines_coarse:
        depth_factor = line[0][1] / frame.shape[0]
        intensity = int(coarse_line_params['intensity'] * (1 - depth_factor * coarse_line_params['depth_factor']))
        
        for i in range(len(line) - 1):
            pt1 = tuple(map(int, line[i]))
            pt2 = tuple(map(int, line[i + 1]))
            cv2.line(coarse_mask, pt1, pt2, 255, coarse_line_params['thickness'])
    
    # Remove grid where it intersects with ground plane
    fine_mask = cv2.bitwise_and(fine_mask, cv2.bitwise_not(ground_plane_mask))
    coarse_mask = cv2.bitwise_and(coarse_mask, cv2.bitwise_not(ground_plane_mask))
    
    # Draw the fine grid in white
    output[fine_mask > 0] = [255, 255, 255]  # White color for fine lines
    
    # Draw the coarse grid in yellow
    output[coarse_mask > 0] = [0, 255, 255]  # Yellow color for coarse lines
    
    # Draw depth markers with increased visibility
    for i, line in enumerate(h_lines_coarse[::2]):  # Every other line
        if len(line) > 0:
            x = line[0][0]
            y = line[0][1]
            depth_meters = i * grid_size * 2  # Every other line, so multiply by 2
            cv2.putText(output, f"{depth_meters}m", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend with original frame with increased alpha
    alpha = 0.5
    cv2.addWeighted(output, alpha, frame, 1 - alpha, 0, output)
    
    return output

def create_debug_view(frame, fg_mask, bg_model, grid_points, h_lines_coarse, v_lines_coarse, 
                    h_lines_fine, v_lines_fine, stable_markers, pose_detector):
    """
    Create a debug view showing original frame, background model, and foreground mask
    with grid overlay and intersection highlights
    """
    # Get dimensions
    height, width = frame.shape[:2]
    
    # Create binary grid mask
    grid_mask = create_binary_grid_mask(fg_mask.shape, grid_points, h_lines_coarse, v_lines_coarse)
    
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
    frame_with_grid = draw_masked_grid(frame.copy(), grid_points, h_lines_coarse, v_lines_coarse, 
                                    h_lines_fine, v_lines_fine, stable_markers, fg_mask, 
                                    intersection_mask, grid_size=0.5)
    
    # Add pose detection to the frame
    frame_with_poses = pose_detector.draw_poses(frame_with_grid)
    
    # Draw grid on background model (without removing intersections)
    bg_with_grid = draw_masked_grid(bg_display, grid_points, h_lines_coarse, v_lines_coarse, 
                                    h_lines_fine, v_lines_fine, stable_markers, np.zeros_like(fg_mask), 
                                    np.zeros_like(fg_mask), grid_size=0.5)
    
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
    Detect orange markers with multiple HSV ranges to handle lighting variations.
    Returns markers in order: [0,0], [5,0], [5,5], [0,5] in real-world coordinates.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create debug visualization
    debug_frame = frame.copy()
    
    hsv_ranges = [
        (np.array([0, 150, 150]), np.array([20, 255, 255])),
        (np.array([0, 100, 100]), np.array([20, 255, 200])),
        (np.array([0, 50, 200]), np.array([25, 150, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ]
    
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for i, (lower, upper) in enumerate(hsv_ranges):
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Save individual HSV range masks for debugging
        cv2.imwrite(f'hsv_mask_{i}.jpg', mask)
    
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save the combined mask
    cv2.imwrite('combined_mask.jpg', combined_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marker_centers = []
    min_area = 100
    max_area = 10000
    
    # Draw all detected contours for debugging
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('all_contours.jpg', contour_frame)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
            
            if circularity > 0.6:
                marker_centers.append((int(x), int(y)))
                # Draw detected marker on debug frame
                cv2.circle(debug_frame, (int(x), int(y)), 10, (0, 165, 255), -1)
    
    # Save debug frame with all detected markers
    cv2.imwrite('detected_markers.jpg', debug_frame)
    
    if len(marker_centers) == 4:
        # Sort by x-coordinate first (left to right)
        marker_centers.sort(key=lambda p: p[0])
        # Split into left and right pairs
        left_points = sorted(marker_centers[:2], key=lambda p: p[1])  # Sort by y (top to bottom)
        right_points = sorted(marker_centers[2:], key=lambda p: p[1])  # Sort by y (top to bottom)
        
        # Combine in order: [0,0], [5,0], [5,5], [0,5]
        # left_points[1] is bottom left [0,0]
        # right_points[1] is bottom right [5,0]
        # right_points[0] is top right [5,5]
        # left_points[0] is top left [0,5]
        marker_centers = [left_points[1], right_points[1], right_points[0], left_points[0]]
        
        return np.array(marker_centers)
    
    return None

def draw_grid(frame, grid_width, grid_height, cell_size):
    """Draw a grid on the frame with depth indicators."""
    # Create a copy of the frame for the grid
    grid_frame = frame.copy()
    
    # Define colors for different depth levels
    colors = {
        'near': (200, 200, 200),    # Brighter for closer
        'mid': (150, 150, 150),     # Medium for middle distance
        'far': (100, 100, 100)      # Darker for farther
    }
    
    # Draw horizontal lines with varying intensity based on depth
    for y in range(0, grid_height, cell_size):
        # Calculate depth factor (0 to 1) based on y position
        depth_factor = y / grid_height
        
        # Choose color based on depth
        if depth_factor < 0.33:
            color = colors['near']
        elif depth_factor < 0.66:
            color = colors['mid']
        else:
            color = colors['far']
            
        cv2.line(grid_frame, (0, y), (grid_width, y), color, 1)
        
        # Add depth markers every 5 cells
        if y % (cell_size * 5) == 0:
            depth_text = f"{int(depth_factor * 10)}m"
            cv2.putText(grid_frame, depth_text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw vertical lines
    for x in range(0, grid_width, cell_size):
        cv2.line(grid_frame, (x, 0), (x, grid_height), (100, 100, 100), 1)
    
    # Add depth scale on the right side
    scale_height = grid_height
    scale_width = 30
    scale_x = grid_width - scale_width - 10
    
    # Draw depth scale gradient
    for y in range(scale_height):
        depth_factor = y / scale_height
        color = (
            int(200 * (1 - depth_factor)),  # Brighter to darker
            int(200 * (1 - depth_factor)),
            int(200 * (1 - depth_factor))
        )
        cv2.line(grid_frame, 
                (scale_x, y), 
                (scale_x + scale_width, y), 
                color, 1)
    
    # Add depth labels
    cv2.putText(grid_frame, "Near", (scale_x, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['near'], 1)
    cv2.putText(grid_frame, "Far", (scale_x, scale_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['far'], 1)
    
    # Blend the grid with the original frame
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(grid_frame, alpha, frame, 1 - alpha, 0, frame)

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
        try:
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
        except (IndexError, ValueError):
            # If there's any error accessing keypoints, return previous positions
            return prev_ball_carrier, prev_tackler
    
    # If no previous positions, use leftmost and rightmost poses
    try:
        left_hip_x = [pose[11][0] for pose in poses]
        ball_carrier_idx = np.argmin(left_hip_x)  # Leftmost is ball carrier
        tackler_idx = np.argmax(left_hip_x)  # Rightmost is tackler
        
        return poses[ball_carrier_idx], poses[tackler_idx]
    except (IndexError, ValueError):
        # If there's any error accessing keypoints, return None
        return None, None

def draw_annotations(frame, ball_carrier, tackler, ball_carrier_history, tackler_history, calibration_data):
    colors = {
        'ball_carrier': (0, 255, 0),    # Green
        'tackler': (0, 0, 255),         # Red
        'measurement': (0, 255, 255),   # Yellow for distance
        'coordinates': (255, 255, 255), # White for coordinates
        'grid': (100, 100, 100),        # Gray
        'history': (255, 255, 255)      # White for history
    }
    
    # Define the skeleton connections
    skeleton = [
        # Face connections
        [0, 1], [0, 2],  # nose to eyes
        [1, 3], [2, 4],  # eyes to ears
        
        # Upper body
        [5, 6],          # shoulders
        [5, 7], [7, 9],  # left arm
        [6, 8], [8, 10], # right arm
        [5, 11], [6, 12], # shoulders to hips
        
        # Lower body
        [11, 12],        # hips
        [11, 13], [13, 15], # left leg
        [12, 14], [14, 16]  # right leg
    ]
    
    def draw_skeleton(keypoints, color, is_ball_carrier=True):
        if keypoints is None:
            return None, None, None
            
        # Draw keypoints
        for kp in keypoints:
            # Check if keypoint has confidence score
            if len(kp) > 2 and kp[2] > 0.5:  # Only draw if confidence > 0.5
                x, y = map(int, kp[:2])
                cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw skeleton connections
        for connection in skeleton:
            start_idx, end_idx = connection
            if (len(keypoints[start_idx]) > 2 and keypoints[start_idx][2] > 0.5 and 
                len(keypoints[end_idx]) > 2 and keypoints[end_idx][2] > 0.5):
                start_point = tuple(map(int, keypoints[start_idx][:2]))
                end_point = tuple(map(int, keypoints[end_idx][:2]))
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Calculate key measurements
        if is_ball_carrier:
            # Get wrist positions (for ball carrier)
            left_wrist = keypoints[9] if len(keypoints[9]) > 2 and keypoints[9][2] > 0.5 else None
            right_wrist = keypoints[10] if len(keypoints[10]) > 2 and keypoints[10][2] > 0.5 else None
            return left_wrist, right_wrist, None
        else:
            # Get shoulder positions (for tackler)
            left_shoulder = keypoints[5] if len(keypoints[5]) > 2 and keypoints[5][2] > 0.5 else None
            right_shoulder = keypoints[6] if len(keypoints[6]) > 2 and keypoints[6][2] > 0.5 else None
            return left_shoulder, right_shoulder, None
    
    # Draw skeletons and get key points
    ball_carrier_points = draw_skeleton(ball_carrier, colors['ball_carrier'], True)
    tackler_points = draw_skeleton(tackler, colors['tackler'], False)
    
    current_distance = None
    
    # Calculate and display real-world measurements if both players are detected
    if ball_carrier is not None and tackler is not None:
        try:
            # Get the transform matrix from calibration data
            transform_matrix = calibration_data['transform_matrix']
            
            def project_hip_to_ground(hip_pos, ankle_pos):
                """Project hip position onto ground plane using ankle position"""
                # Get hip and ankle positions
                hip_x, hip_y = hip_pos[:2]
                ankle_x, ankle_y = ankle_pos[:2]
                
                # Calculate the vertical line from hip to ground
                # This is the intersection of the vertical line with the ground plane
                ground_x = hip_x
                ground_y = ankle_y  # Use ankle y as ground level
                
                return np.array([ground_x, ground_y])
            
            # Get hip and ankle positions for both players
            ball_carrier_hip = np.mean([ball_carrier[11], ball_carrier[12]], axis=0)  # Average of left and right hips
            ball_carrier_ankle = np.mean([ball_carrier[15], ball_carrier[16]], axis=0)  # Average of left and right ankles
            
            tackler_hip = np.mean([tackler[11], tackler[12]], axis=0)
            tackler_ankle = np.mean([tackler[15], tackler[16]], axis=0)
            
            # Project hips onto ground plane
            ball_carrier_ground = project_hip_to_ground(ball_carrier_hip, ball_carrier_ankle)
            tackler_ground = project_hip_to_ground(tackler_hip, tackler_ankle)
            
            # Convert ground positions to real-world coordinates
            ball_carrier_ground_h = np.concatenate([ball_carrier_ground, [1]])
            tackler_ground_h = np.concatenate([tackler_ground, [1]])
            
            # Apply perspective transform to get real-world coordinates
            ball_carrier_real = np.dot(ball_carrier_ground_h, transform_matrix.T)
            tackler_real = np.dot(tackler_ground_h, transform_matrix.T)
            
            # Normalize homogeneous coordinates
            ball_carrier_real = ball_carrier_real[:2] / ball_carrier_real[2]
            tackler_real = tackler_real[:2] / tackler_real[2]
            
            # Ensure coordinates stay within grid bounds [0,5] for both x and y
            def clamp_coordinates(coords):
                return np.clip(coords, 0, 5)
            
            ball_carrier_real = clamp_coordinates(ball_carrier_real)
            tackler_real = clamp_coordinates(tackler_real)
            
            # Calculate distance in meters using real-world coordinates
            dist_meters = np.linalg.norm(ball_carrier_real - tackler_real)
            
            # Ensure distance doesn't exceed grid size
            dist_meters = min(dist_meters, 5.0)  # Maximum distance is 5 meters
            current_distance = dist_meters
            
            # Log if coordinates were clamped
            if np.any(ball_carrier_real == 0) or np.any(ball_carrier_real == 5):
                logging.warning(f"Ball carrier coordinates were clamped: {ball_carrier_real}")
            if np.any(tackler_real == 0) or np.any(tackler_real == 5):
                logging.warning(f"Tackler coordinates were clamped: {tackler_real}")
            
            # Draw distance line in pixel coordinates (now using ground positions)
            cv2.line(frame, 
                    tuple(map(int, ball_carrier_ground)), 
                    tuple(map(int, tackler_ground)), 
                    colors['measurement'], 3)  # Yellow line
            
            # Prepare all text elements
            dist_text = f"Distance: {dist_meters:.2f}m"
            bc_coord_text = f"BC: ({ball_carrier_real[0]:.1f}, {ball_carrier_real[1]:.1f})"
            t_coord_text = f"T: ({tackler_real[0]:.1f}, {tackler_real[1]:.1f})"
            
            # Calculate text sizes
            font_scale = 0.8  # Increased font size
            thickness = 2
            dist_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            bc_size = cv2.getTextSize(bc_coord_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            t_size = cv2.getTextSize(t_coord_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Calculate positions for all text elements
            text_y = 50  # 50 pixels from top
            
            # Center the distance text
            dist_x = (frame.shape[1] - dist_size[0]) // 2
            
            # Position coordinate texts on either side
            padding = 20  # Space between texts
            bc_x = dist_x - bc_size[0] - padding
            t_x = dist_x + dist_size[0] + padding
            
            # Draw all text elements
            cv2.putText(frame, 
                       dist_text, 
                       (dist_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       colors['measurement'], 
                       thickness)
            
            cv2.putText(frame, 
                       bc_coord_text, 
                       (bc_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       colors['coordinates'], 
                       thickness)
            
            cv2.putText(frame, 
                       t_coord_text, 
                       (t_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       colors['coordinates'], 
                       thickness)
            
            # Draw distance history (only lines, no labels)
            if len(ball_carrier_history) > 0 and len(tackler_history) > 0:
                # Draw lines connecting previous positions
                for i in range(len(ball_carrier_history) - 1):
                    prev_bc_hip = np.mean([ball_carrier_history[i][11], ball_carrier_history[i][12]], axis=0)
                    prev_t_hip = np.mean([tackler_history[i][11], tackler_history[i][12]], axis=0)
                    curr_bc_hip = np.mean([ball_carrier_history[i+1][11], ball_carrier_history[i+1][12]], axis=0)
                    curr_t_hip = np.mean([tackler_history[i+1][11], tackler_history[i+1][12]], axis=0)
                    
                    # Draw lines with decreasing opacity
                    alpha = 0.3 * (i / len(ball_carrier_history))
                    cv2.line(frame, 
                            tuple(map(int, prev_bc_hip[:2])), 
                            tuple(map(int, curr_bc_hip[:2])), 
                            colors['ball_carrier'], 2)  # Green for ball carrier
                    cv2.line(frame, 
                            tuple(map(int, prev_t_hip[:2])), 
                            tuple(map(int, curr_t_hip[:2])), 
                            colors['tackler'], 2)  # Red for tackler
            
            # Log the distance and coordinates for debugging
            logging.info(f"Player distance: {dist_meters:.2f}m")
            logging.info(f"Ball carrier real-world position: {ball_carrier_real}")
            logging.info(f"Tackler real-world position: {tackler_real}")
            
        except Exception as e:
            logging.warning(f"Could not calculate real-world distance: {str(e)}")
    
    return ball_carrier_points, tackler_points, current_distance

def calibrate_markers(video_path, num_frames=50):
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
    
    logging.info("Starting marker calibration...")
    
    # First pass: collect marker positions
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Use robust orange marker detection
        detected_markers = detect_orange_markers(frame)
        if detected_markers is not None and len(detected_markers) == 4:
            marker_positions.append(detected_markers)
            
            # Create visualization of detected markers
            vis_frame = frame.copy()
            for marker in detected_markers:
                cv2.circle(vis_frame, tuple(marker), 10, (0, 165, 255), -1)
            
            # Save frame with detected markers
            cv2.imwrite(f'marker_detection_{frame_count:03d}.jpg', vis_frame)
        
        frame_count += 1
    
    if len(marker_positions) == 0:
        raise ValueError("No valid marker sets detected during calibration")
    
    # Convert to numpy array for easier processing
    marker_positions = np.array(marker_positions)
    
    # Calculate stability metrics for each marker
    stability_threshold = 20  # pixels
    stable_frames = []
    
    for i in range(len(marker_positions)):
        # Calculate distances between consecutive positions
        if i > 0:
            distances = np.linalg.norm(marker_positions[i] - marker_positions[i-1], axis=1)
            # Check if all markers are stable
            if np.all(distances < stability_threshold):
                stable_frames.append(i)
    
    logging.info(f"Found {len(stable_frames)} stable frames out of {len(marker_positions)} total frames")
    
    if len(stable_frames) < 5:  # Require at least 5 stable frames
        raise ValueError("Insufficient stable marker positions detected")
    
    # Use only stable frames for averaging
    stable_positions = marker_positions[stable_frames]
    corners = np.mean(stable_positions, axis=0)
    
    # Calculate standard deviation of marker positions
    std_dev = np.std(stable_positions, axis=0)
    logging.info(f"Marker position standard deviation: {std_dev}")
    
    # Create visualization of final marker positions
    if len(stable_frames) > 0:
        final_frame = cv2.imread(f'marker_detection_{stable_frames[-1]:03d}.jpg')
        if final_frame is not None:
            for corner in corners:
                cv2.circle(final_frame, tuple(map(int, corner)), 10, (0, 165, 255), -1)
            
            cv2.imwrite('final_marker_positions.jpg', final_frame)
    
    # Create grid points with proper perspective
    grid_points, h_lines_coarse, v_lines_coarse, h_lines_fine, v_lines_fine, transform_matrix, inverse_matrix = create_grid_points(corners)
    
    # Create visualization of the last frame with grid and poses
    if final_frame is not None:
        # Draw grid with proper perspective
        grid_frame = draw_masked_grid(final_frame, grid_points, h_lines_coarse, v_lines_coarse, 
                                    h_lines_fine, v_lines_fine, [], 
                                    np.zeros_like(final_frame[:,:,0]), 
                                    np.zeros_like(final_frame[:,:,0]), 
                                    grid_size=0.5)
        
        # Save the visualization
        cv2.imwrite('calibration_visualization.jpg', grid_frame)
        logging.info("Saved calibration visualization to calibration_visualization.jpg")
    
    logging.info("Marker calibration completed successfully")
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'grid_points': grid_points,
        'h_lines_coarse': h_lines_coarse,
        'v_lines_coarse': v_lines_coarse,
        'h_lines_fine': h_lines_fine,
        'v_lines_fine': v_lines_fine,
        'corners': corners,
        'transform_matrix': transform_matrix,
        'inverse_matrix': inverse_matrix,
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
    grid_points = calibration_data['grid_points']
    h_lines_coarse = calibration_data['h_lines_coarse']
    v_lines_coarse = calibration_data['v_lines_coarse']
    h_lines_fine = calibration_data['h_lines_fine']
    v_lines_fine = calibration_data['v_lines_fine']
    
    # Initialize pose tracking variables
    ball_carrier_history = calibration_data['ball_carrier_history']
    tackler_history = calibration_data['tackler_history']
    prev_ball_carrier = calibration_data['prev_ball_carrier']
    prev_tackler = calibration_data['prev_tackler']
    
    # Prepare output video
    output_path = 'rugby_analysis_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize distance tracking
    distances = []
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Create a copy of the frame for drawing
        output_frame = frame.copy()
            
        # Process frame for pose detection
        results = model(frame)
        if len(results[0].keypoints) >= 1:
            ball_carrier, tackler = identify_players(results[0].keypoints, prev_ball_carrier, prev_tackler)
            if ball_carrier is not None and tackler is not None:
                prev_ball_carrier, prev_tackler = ball_carrier, tackler
                ball_carrier_history.append(ball_carrier)
                tackler_history.append(tackler)
                
                # Draw pose annotations and get distance
                ball_carrier_points, tackler_points, current_distance = draw_annotations(
                    output_frame, ball_carrier, tackler, ball_carrier_history, tackler_history, calibration_data)
                
                # Store distance if available
                if current_distance is not None:
                    distances.append((frame_count, current_distance))
                    logging.info(f"Frame {frame_count}: Distance = {current_distance:.2f}m")
        
        # Draw perspective grid after pose annotations
        output_frame = draw_masked_grid(
            output_frame,
            grid_points,
            h_lines_coarse,
            v_lines_coarse,
            h_lines_fine,
            v_lines_fine,
            [],
            np.zeros((height, width), dtype=np.uint8),
            np.zeros((height, width), dtype=np.uint8),
            grid_size=0.5
        )
        
        # Write frame to output video
        out.write(output_frame)
        
        # Save every 10th frame for verification
        if frame_count % 10 == 0:
            cv2.imwrite(f'frame_{frame_count:03d}.jpg', output_frame)
            logging.info(f"Saved verification frame {frame_count}")
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Save distance data to CSV
    if distances:
        import csv
        with open('distance_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Distance (m)'])
            writer.writerows(distances)
        logging.info(f"Saved distance data to distance_data.csv")
    
    logging.info(f"Processed {frame_count} frames")
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