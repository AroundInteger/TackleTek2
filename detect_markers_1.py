import cv2
import numpy as np
import logging

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
    min_area = 100  # Adjust based on your video
    max_area = 10000  # Adjust based on your video
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Use minEnclosingCircle for better center estimation
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
            
            # Check if the contour is roughly circular (circularity > 0.6)
            if circularity > 0.6:
                marker_centers.append((int(x), int(y)))
    
    # If we found 4 markers, sort them spatially
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

# Main processing loop modification
def process_frame(frame):
    """
    Process a single frame
    """
    # Detect markers
    markers = detect_orange_markers(frame)
    
    # Draw markers and connections
    if markers is not None:
        frame = draw_marker_connections(frame, markers)
    
    return frame

# Example usage in your main loop:
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Process frame for marker detection
    annotated_frame = process_frame(frame.copy())
    
    # Your existing pose estimation code here
    results = model(frame)
    pose_frame = results[0].plot()
    
    # Combine the annotations
    final_frame = cv2.addWeighted(pose_frame, 0.7, annotated_frame, 0.3, 0)
    
    # Write and display
    out.write(final_frame)
    cv2.imshow("Rugby Tackle Analysis", final_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break