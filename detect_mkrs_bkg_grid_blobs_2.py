#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 2024
@author: rowanbrown
Modified to remove ball detection and add grid to all panels
Added YOLOv8 pose detection
Modified to organize output files in structured folders
"""

import cv2
import numpy as np
import logging
import os
import sys
from collections import defaultdict, deque
from ultralytics import YOLO
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize YOLO model
model = YOLO('yolov8x-pose.pt')  # Using the most accurate model

# Global variable to store output directories
OUTPUT_DIRS = None

# Create output directory structure
def create_output_directories():
    """
    Creates the output directory structure for organized file storage.
    
    Returns
    -------
    dict
        Dictionary containing paths to all output directories.
    """
    # Create timestamp for unique output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"rugby_analysis_output_{timestamp}"
    
    # Define directory structure
    directories = {
        'base': base_output_dir,
        'masks': os.path.join(base_output_dir, 'masks'),
        'contours': os.path.join(base_output_dir, 'contours'),
        'markers': os.path.join(base_output_dir, 'markers'),
        'calibration': os.path.join(base_output_dir, 'calibration'),
        'frames': os.path.join(base_output_dir, 'frames'),
        'videos': os.path.join(base_output_dir, 'videos'),
        'data': os.path.join(base_output_dir, 'data')
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    return directories

def ensure_output_dirs():
    """
    Ensures output directories are created if they don't exist.
    This is called before any file operations.
    """
    global OUTPUT_DIRS
    if OUTPUT_DIRS is None:
        OUTPUT_DIRS = create_output_directories()
    return OUTPUT_DIRS

class PlayerTracker:
    """
    Tracks and smooths the position of a player using a Kalman filter and temporal averaging.

    Attributes
    ----------
    max_history : int
        Maximum number of positions to keep in history for smoothing.
    position_history : deque
        History of player positions.
    y_smooth_history : deque
        History of Y-coordinates for additional smoothing.
    kalman : cv2.KalmanFilter
        Kalman filter instance for position prediction and correction.

    Methods
    -------
    update(position)
        Updates the tracker with a new position and returns the smoothed position.
    """
    def __init__(self, max_history=30):
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.y_smooth_history = deque(maxlen=max_history)
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        
        # Initialize Kalman filter parameters
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.03
        
        # Increase process noise for y-direction to allow for more movement
        self.kalman.processNoiseCov[1,1] = 0.01  # Reduce y-direction noise
        
        # Initialize state
        self.kalman.statePre = np.array([[0], [0], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[0], [0], [0], [0]], np.float32)
        
    def update(self, position):
        if position is None:
            return None
            
        # Update position history
        self.position_history.append(position)
        
        # Apply Kalman filter
        measurement = np.array([[position[0]], [position[1]]], np.float32)
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        
        # Get filtered position
        filtered_pos = np.array([prediction[0,0], prediction[1,0]])
        
        # Apply temporal smoothing to Y-coordinate
        if len(self.y_smooth_history) > 0:
            # Calculate weighted average of Y-coordinates
            weights = np.linspace(0.1, 1.0, len(self.y_smooth_history))
            y_smooth = np.average([p[1] for p in self.y_smooth_history], weights=weights)
            
            # Blend current Y with smoothed Y (70% current, 30% smoothed)
            filtered_pos[1] = 0.7 * filtered_pos[1] + 0.3 * y_smooth
        
        # Update smooth history
        self.y_smooth_history.append(filtered_pos)
        
        return filtered_pos

def create_euclidean_grid_overlay(frame, ball_carrier_pos, tackler_pos, 
                                  ball_carrier_history, tackler_history,
                                  grid_size=0.25, overlay_size=600):
    """
    Draws a fixed-size (5x5m) Euclidean grid overlay in the top-right corner of the frame,
    showing the real-world positions and distance of the two tracked players.

    Parameters
    ----------
    frame : np.ndarray
        The video frame to annotate.
    ball_carrier_pos : array-like or None
        Real-world (x, y) position of the ball carrier.
    tackler_pos : array-like or None
        Real-world (x, y) position of the tackler.
    ball_carrier_history : deque
        History of ball carrier positions.
    tackler_history : deque
        History of tackler positions.
    grid_size : float, optional
        Size of each grid cell in meters (default is 0.25).
    overlay_size : int, optional
        Size of the overlay in pixels (default is 600).

    Returns
    -------
    frame : np.ndarray
        The frame with the Euclidean grid overlay.
    """
    # Create overlay image
    overlay = np.zeros((overlay_size, overlay_size, 3), dtype=np.uint8)
    
    # Calculate scale factor (pixels per meter)
    scale = overlay_size / 5.0  # 5 meters total width
    
    # Draw only coarse grid lines (0.5m intervals) with reduced thickness
    for i in range(11):  # 0.5m intervals (11 lines for 5m)
        pos = int(i * scale * 0.5)
        # Draw vertical lines
        cv2.line(overlay, (pos, 0), (pos, overlay_size), (0, 255, 255), 1)  # Yellow coarse lines, thickness=1
        # Draw horizontal lines
        cv2.line(overlay, (0, pos), (overlay_size, pos), (0, 255, 255), 1)  # Yellow coarse lines, thickness=1
    
    # Add axis labels and numbers
    # X-axis (bottom)
    for i in range(6):  # 0 to 5 meters
        x = int(i * scale)
        cv2.putText(overlay, f"{i}", (x, overlay_size-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "X (m)", (overlay_size//2, overlay_size-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Y-axis (left)
    for i in range(6):  # 0 to 5 meters
        y = overlay_size - int(i * scale)
        cv2.putText(overlay, f"{i}", (5, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "Y (m)", (5, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Comment out trajectory drawing for now
    """
    # Draw trajectory lines
    def draw_trajectory(history, color, max_points=10):
        if len(history) < 2:
            return
        history_list = list(history)
        if len(history_list) > max_points:
            history_list = history_list[-max_points:]
        points = []
        for pos in history_list:
            x = int(pos[0] * scale)
            y = overlay_size - int(pos[1] * scale)  # Flip Y coordinate to match real-world grid
            points.append((x, y))
        for i in range(len(points) - 1):
            cv2.line(overlay, points[i], points[i+1], color, 1)  # Thickness=1 for trajectories
    
    # Draw trajectories
    draw_trajectory(ball_carrier_history, (0, 255, 0))  # Green for ball carrier
    draw_trajectory(tackler_history, (0, 0, 255))       # Red for tackler
    """
    
    # Draw current positions with representative player radii
    if ball_carrier_pos is not None and tackler_pos is not None:
        # Calculate positions for both players
        bc_x = int(ball_carrier_pos[0] * scale)
        bc_y = overlay_size - int(ball_carrier_pos[1] * scale)
        t_x = int(tackler_pos[0] * scale)
        t_y = overlay_size - int(tackler_pos[1] * scale)
        
        # Draw connecting line between player centers
        cv2.line(overlay, (bc_x, bc_y), (t_x, t_y), (0, 255, 255), 2)  # Yellow line, thickness=2
        
        # Draw ball carrier
        player_radius = int(0.4 * scale)  # Convert meters to pixels
        cv2.circle(overlay, (bc_x, bc_y), player_radius, (0, 255, 0), 2)  # Green circle
        coord_text = f"BC: ({ball_carrier_pos[0]:.1f}, {ball_carrier_pos[1]:.1f})"
        cv2.putText(overlay, coord_text, (bc_x + player_radius + 5, bc_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw tackler
        cv2.circle(overlay, (t_x, t_y), player_radius, (0, 0, 255), 2)  # Red circle
        coord_text = f"T: ({tackler_pos[0]:.1f}, {tackler_pos[1]:.1f})"
        cv2.putText(overlay, coord_text, (t_x + player_radius + 5, t_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate and display distance
        dist = np.linalg.norm(np.array(ball_carrier_pos) - np.array(tackler_pos))
        # Position distance text at midpoint of connecting line
        mid_x = (bc_x + t_x) // 2
        mid_y = (bc_y + t_y) // 2
        cv2.putText(overlay, f"{dist:.2f}m", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add border
    cv2.rectangle(overlay, (0, 0), (overlay_size-1, overlay_size-1), (255, 255, 255), 1)
    
    # Add title with larger font
    cv2.putText(overlay, "Euclidean Grid", (20, overlay_size-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Place overlay in top-right corner of frame
    frame[10:10+overlay_size, frame.shape[1]-overlay_size-10:frame.shape[1]-10] = overlay
    
    return frame

def draw_annotations(frame, ball_carrier, tackler, ball_carrier_history, tackler_history, calibration_data):
    """
    Draws pose skeletons, real-world coordinates, and distance annotations for the two players.
    Also overlays the Euclidean grid.

    Parameters
    ----------
    frame : np.ndarray
        The video frame to annotate.
    ball_carrier : np.ndarray
        Keypoints for the ball carrier.
    tackler : np.ndarray
        Keypoints for the tackler.
    ball_carrier_history : deque
        History of ball carrier positions.
    tackler_history : deque
        History of tackler positions.
    calibration_data : dict
        Calibration data including perspective transform and grid info.

    Returns
    -------
    ball_carrier_points, tackler_points, current_distance, ball_carrier_pos, tackler_pos
        Various annotation and measurement results for further processing.
    """
    colors = {
        'ball_carrier': (0, 255, 0),    # Green
        'tackler': (0, 0, 255),         # Red
        'measurement': (0, 255, 255),   # Yellow for distance
        'coordinates': (255, 255, 255), # White for coordinates
        'grid': (100, 100, 100),        # Gray
        'history': (255, 255, 255)      # White for history
    }
    
    # Initialize trackers if they don't exist
    if not hasattr(draw_annotations, 'ball_carrier_tracker'):
        draw_annotations.ball_carrier_tracker = PlayerTracker()
        draw_annotations.tackler_tracker = PlayerTracker()
    
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
    
    # Initialize real-world positions
    ball_carrier_pos = None
    tackler_pos = None
    current_distance = None
    
    # Calculate and display real-world measurements if both players are detected
    if ball_carrier is not None and tackler is not None:
        try:
            # Get the transform matrix from calibration data
            transform_matrix = calibration_data['transform_matrix']
            
            def project_hip_to_ground(hip_pos, ankle_pos):
                """Project hip position onto ground plane using ankle position"""
                hip_x, hip_y = hip_pos[:2]
                ankle_x, ankle_y = ankle_pos[:2]
                ground_x = hip_x
                ground_y = ankle_y  # Use ankle y as ground level
                return np.array([ground_x, ground_y])
            
            # Get hip and ankle positions for both players
            ball_carrier_hip = np.mean([ball_carrier[11], ball_carrier[12]], axis=0)
            ball_carrier_ankle = np.mean([ball_carrier[15], ball_carrier[16]], axis=0)
            
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
            
            # Apply tracking and smoothing
            ball_carrier_pos = draw_annotations.ball_carrier_tracker.update(ball_carrier_real)
            tackler_pos = draw_annotations.tackler_tracker.update(tackler_real)
            
            # Ensure coordinates stay within grid bounds [0,5] for both x and y
            def clamp_coordinates(coords):
                return np.clip(coords, 0, 5)
            
            ball_carrier_pos = clamp_coordinates(ball_carrier_pos)
            tackler_pos = clamp_coordinates(tackler_pos)
            
            # Calculate distance in meters using real-world coordinates
            dist_meters = np.linalg.norm(ball_carrier_pos - tackler_pos)
            current_distance = min(dist_meters, 5.0)  # Maximum distance is 5 meters
            
            # Draw distance line in pixel coordinates (now using ground positions)
            cv2.line(frame, 
                    tuple(map(int, ball_carrier_ground)), 
                    tuple(map(int, tackler_ground)), 
                    colors['measurement'], 3)  # Yellow line
            
            # Prepare all text elements
            dist_text = f"Distance: {dist_meters:.2f}m"
            bc_coord_text = f"BC: ({ball_carrier_pos[0]:.1f}, {ball_carrier_pos[1]:.1f})"
            t_coord_text = f"T: ({tackler_pos[0]:.1f}, {tackler_pos[1]:.1f})"
            
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
                    prev_bc_hip = np.mean([ball_carrier_history[i][0], ball_carrier_history[i][1]])
                    prev_t_hip = np.mean([tackler_history[i][0], tackler_history[i][1]])
                    curr_bc_hip = np.mean([ball_carrier_history[i+1][0], ball_carrier_history[i+1][1]])
                    curr_t_hip = np.mean([tackler_history[i+1][0], tackler_history[i+1][1]])
                    
                    # Draw lines with decreasing opacity
                    alpha = 0.3 * (i / len(ball_carrier_history))
                    cv2.line(frame, 
                            tuple(map(int, prev_bc_hip)), 
                            tuple(map(int, curr_bc_hip)), 
                            colors['ball_carrier'], 2)  # Green for ball carrier
                    cv2.line(frame, 
                            tuple(map(int, prev_t_hip)), 
                            tuple(map(int, curr_t_hip)), 
                            colors['tackler'], 2)  # Red for tackler
            
            # Log the distance and coordinates for debugging
            logging.info(f"Player distance: {dist_meters:.2f}m")
            logging.info(f"Ball carrier real-world position: {ball_carrier_pos}")
            logging.info(f"Tackler real-world position: {tackler_pos}")
            
        except Exception as e:
            logging.warning(f"Could not calculate real-world distance: {str(e)}")
    
    # Add Euclidean grid overlay
    frame = create_euclidean_grid_overlay(frame, ball_carrier_pos, tackler_pos,
                                        ball_carrier_history, tackler_history)
    
    return ball_carrier_points, tackler_points, current_distance, ball_carrier_pos, tackler_pos

def process_video(video_path, calibration_data):
    """
    Processes the input video, performing pose detection, player tracking, grid overlay,
    and distance measurement for each frame. Saves annotated video and distance data.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    calibration_data : dict
        Calibration data including grid points and perspective transform.

    Returns
    -------
    None
    """
    # Ensure output directories exist
    output_dirs = ensure_output_dirs()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get video properties from calibration
    grid_points = calibration_data['grid_points']
    h_lines_coarse = calibration_data['h_lines_coarse']
    v_lines_coarse = calibration_data['v_lines_coarse']
    h_lines_fine = calibration_data['h_lines_fine']
    v_lines_fine = calibration_data['v_lines_fine']
    
    # Initialize pose tracking variables
    ball_carrier_history = deque(maxlen=30)  # Store real-world positions
    tackler_history = deque(maxlen=30)       # Store real-world positions
    prev_ball_carrier = None
    prev_tackler = None
    
    # Prepare output video path
    output_path = os.path.join(output_dirs['videos'], 'rugby_analysis_output_2.mp4')
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
                
                # Draw pose annotations and get distance
                ball_carrier_points, tackler_points, current_distance, ball_carrier_pos, tackler_pos = draw_annotations(
                    output_frame, ball_carrier, tackler, ball_carrier_history, tackler_history, calibration_data)
                
                # Store real-world positions in history
                if ball_carrier_pos is not None:
                    ball_carrier_history.append(ball_carrier_pos)
                if tackler_pos is not None:
                    tackler_history.append(tackler_pos)
                
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
            frame_path = os.path.join(output_dirs['frames'], f'frame_{frame_count:03d}_2.jpg')
            cv2.imwrite(frame_path, output_frame)
            logging.info(f"Saved verification frame {frame_count}")
        
        frame_count += 1

    cap.release()
    out.release()

    # Save distance data to CSV
    if distances:
        import csv
        csv_path = os.path.join(output_dirs['data'], 'distance_data_2.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Distance (m)'])
            writer.writerows(distances)
        logging.info(f"Saved distance data to {csv_path}")

    logging.info(f"Processed {frame_count} frames")
    logging.info("Analysis video saved successfully to %s", output_path)

def calibrate_markers(video_path, num_frames=50):
    """
    Calibrates the field by detecting orange markers in the first few frames of the video.
    Computes the perspective transform and grid layout.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    num_frames : int, optional
        Number of frames to use for calibration (default is 50).

    Returns
    -------
    calibration_data : dict
        Dictionary containing calibration results, grid points, and transform matrices.
    """
    # Ensure output directories exist
    output_dirs = ensure_output_dirs()
    
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
            marker_path = os.path.join(output_dirs['markers'], f'marker_detection_{frame_count:03d}.jpg')
            cv2.imwrite(marker_path, vis_frame)
        
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
        final_frame_path = os.path.join(output_dirs['markers'], f'marker_detection_{stable_frames[-1]:03d}.jpg')
        final_frame = cv2.imread(final_frame_path)
        if final_frame is not None:
            for corner in corners:
                cv2.circle(final_frame, tuple(map(int, corner)), 10, (0, 165, 255), -1)
            
            final_marker_path = os.path.join(output_dirs['markers'], 'final_marker_positions.jpg')
            cv2.imwrite(final_marker_path, final_frame)
    
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
        calibration_path = os.path.join(output_dirs['calibration'], 'calibration_visualization.jpg')
        cv2.imwrite(calibration_path, grid_frame)
        logging.info(f"Saved calibration visualization to {calibration_path}")
    
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

def detect_orange_markers(frame):
    """
    Detects four orange field markers in the frame using multiple HSV color ranges and contour analysis.

    Parameters
    ----------
    frame : np.ndarray
        The video frame to process.

    Returns
    -------
    marker_centers : np.ndarray or None
        Array of four marker centers in image coordinates, or None if not found.
    """
    # Ensure output directories exist
    output_dirs = ensure_output_dirs()
    
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
        mask_path = os.path.join(output_dirs['masks'], f'hsv_mask_{i}.jpg')
        cv2.imwrite(mask_path, mask)
    
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save the combined mask
    combined_mask_path = os.path.join(output_dirs['masks'], 'combined_mask.jpg')
    cv2.imwrite(combined_mask_path, combined_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marker_centers = []
    min_area = 100
    max_area = 10000
    
    # Draw all detected contours for debugging
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    contour_path = os.path.join(output_dirs['contours'], 'all_contours.jpg')
    cv2.imwrite(contour_path, contour_frame)
    
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
    detected_markers_path = os.path.join(output_dirs['markers'], 'detected_markers.jpg')
    cv2.imwrite(detected_markers_path, debug_frame)
    
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

def calculate_perspective_transform(corners):
    """
    Calculates the perspective transform matrix from image to real-world coordinates.

    Parameters
    ----------
    corners : np.ndarray
        Array of four corner points in image coordinates.

    Returns
    -------
    transform_matrix : np.ndarray
        Matrix for mapping image to real-world coordinates.
    inverse_matrix : np.ndarray
        Matrix for mapping real-world to image coordinates.
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

def create_grid_points(corners, grid_size=0.5):
    """
    Generates grid points and lines for overlaying a perspective-correct field grid.

    Parameters
    ----------
    corners : np.ndarray
        Array of four field corner points in image coordinates.
    grid_size : float, optional
        Grid cell size in meters (default is 0.5).

    Returns
    -------
    grid_points, h_lines_coarse, v_lines_coarse, h_lines_fine, v_lines_fine, transform_matrix, inverse_matrix
        Various grid and transform data for overlay and measurement.
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
        'thickness': 1,  # Reduced from 2 to 1
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
    
    # Draw the fine grid in white (but we'll skip this by not applying the fine_mask)
    # output[fine_mask > 0] = [255, 255, 255]  # Commented out to remove white lines
    
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
    
    return output

def identify_players(keypoints, prev_ball_carrier, prev_tackler):
    """
    Identifies which detected poses correspond to the ball carrier and tackler,
    using previous positions and left/right heuristics.

    Parameters
    ----------
    keypoints : np.ndarray
        Array of detected pose keypoints.
    prev_ball_carrier : np.ndarray or None
        Previous ball carrier keypoints.
    prev_tackler : np.ndarray or None
        Previous tackler keypoints.

    Returns
    -------
    ball_carrier, tackler : np.ndarray or None
        Keypoints for the ball carrier and tackler.
    """
    if keypoints is None or len(keypoints) == 0:
        return prev_ball_carrier, prev_tackler
        
    poses = keypoints.data.cpu().numpy()
    if len(poses) == 0:
        return prev_ball_carrier, prev_tackler
        
    # If we have previous positions, use them to identify players
    if prev_ball_carrier is not None and prev_tackler is not None:
        try:
            # Calculate distances to previous positions using hip positions
            dists_to_ball_carrier = []
            dists_to_tackler = []
            
            for pose in poses:
                # Use hip position (average of left and right hips)
                hip_pos = np.mean([pose[11], pose[12]], axis=0)[:2]  # Only x,y coordinates
                prev_bc_hip = np.mean([prev_ball_carrier[11], prev_ball_carrier[12]], axis=0)[:2]
                prev_t_hip = np.mean([prev_tackler[11], prev_tackler[12]], axis=0)[:2]
                
                dists_to_ball_carrier.append(np.linalg.norm(hip_pos - prev_bc_hip))
                dists_to_tackler.append(np.linalg.norm(hip_pos - prev_t_hip))
            
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
            
            # Additional check: if distances are too large, might be wrong identification
            max_allowed_distance = 200  # pixels
            if dists_to_ball_carrier[ball_carrier_idx] > max_allowed_distance or \
               dists_to_tackler[tackler_idx] > max_allowed_distance:
                # Fall back to left/right identification
                left_hip_x = [np.mean([pose[11], pose[12]], axis=0)[0] for pose in poses]
                ball_carrier_idx = np.argmin(left_hip_x)  # Leftmost is ball carrier
                tackler_idx = np.argmax(left_hip_x)  # Rightmost is tackler
            
            return poses[ball_carrier_idx], poses[tackler_idx]
            
        except (IndexError, ValueError) as e:
            logging.warning(f"Error in player identification: {str(e)}")
            # If there's any error accessing keypoints, return previous positions
            return prev_ball_carrier, prev_tackler
    
    # If no previous positions, use leftmost and rightmost poses
    try:
        # Use hip positions for left/right identification
        left_hip_x = [np.mean([pose[11], pose[12]], axis=0)[0] for pose in poses]
        ball_carrier_idx = np.argmin(left_hip_x)  # Leftmost is ball carrier
        tackler_idx = np.argmax(left_hip_x)  # Rightmost is tackler
        
        return poses[ball_carrier_idx], poses[tackler_idx]
    except (IndexError, ValueError) as e:
        logging.warning(f"Error in initial player identification: {str(e)}")
        # If there's any error accessing keypoints, return None
        return None, None

def main():
    global OUTPUT_DIRS
    
    # Create output directory structure
    OUTPUT_DIRS = create_output_directories()
    
    video_path = "R1_1.mp4"
    try:
        # Calibrate markers and initialize pose detection
        calibration_data = calibrate_markers(video_path)
        # Process the video with calibrated parameters
        process_video(video_path, calibration_data)
        
        logging.info(f"All output files saved to: {OUTPUT_DIRS['base']}")
        
    except Exception as e:
        logging.error("Error processing video: %s", str(e))
        raise

if __name__ == "__main__":
    main() 