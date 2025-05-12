import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLOv8 model
model = YOLO('yolov8x-pose.pt')  # Using the most accurate model

def calculate_speed(prev_pos, curr_pos, fps):
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) * fps

def detect_impact(ball_carrier, tackler, velocity_threshold=2, distance_threshold=50, frame_window=5):
    # Calculate velocities
    ball_carrier_velocity = np.linalg.norm(ball_carrier[-1] - ball_carrier[0]) / (len(ball_carrier) - 1)
    tackler_velocity = np.linalg.norm(tackler[-1] - tackler[0]) / (len(tackler) - 1)
    
    # Check if players are close
    distance = np.linalg.norm(ball_carrier[-1] - tackler[-1])
    
    # Check for sudden velocity change
    if len(ball_carrier) == frame_window and len(tackler) == frame_window:
        ball_carrier_deceleration = ball_carrier_velocity - np.linalg.norm(ball_carrier[-1] - ball_carrier[-2])
        tackler_deceleration = tackler_velocity - np.linalg.norm(tackler[-1] - tackler[-2])
        
        if (distance < distance_threshold and 
            (ball_carrier_deceleration > velocity_threshold or tackler_deceleration > velocity_threshold)):
            return True
    
    return False

def analyze_tackle(ball_carrier, tackler):
    # Check if tackler's head is below ball carrier's wrists
    head_below_wrists = tackler[0][1] > ball_carrier[9][1] and tackler[0][1] > ball_carrier[10][1]
    
    # Check if tackler's head, back, and hips are inline
    head_back_hips_inline = abs(tackler[0][0] - tackler[5][0]) < 20 and abs(tackler[0][0] - tackler[11][0]) < 20
    
    return head_below_wrists, head_back_hips_inline

def draw_labels_and_indicators(frame, ball_carrier, tackler, head_below_wrists):
    # Label Ball Carrier
    cv2.putText(frame, "Ball Carrier", (int(ball_carrier[0][0]), int(ball_carrier[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Label Tackler
    cv2.putText(frame, "Tackler", (int(tackler[0][0]), int(tackler[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw line for ball carrier's wrist height
    wrist_height = min(ball_carrier[9][1], ball_carrier[10][1])
    cv2.line(frame, (0, int(wrist_height)), (frame.shape[1], int(wrist_height)), (0, 255, 255), 1)
    
    # Indicate if tackler's head is below ball carrier's wrists
    color = (0, 255, 0) if head_below_wrists else (0, 0, 255)
    cv2.circle(frame, (int(tackler[0][0]), int(tackler[0][1])), 5, color, -1)

# Open the video file
video_path = "R1_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare output video
output_path = 'analyzed_rugby_tackle_R1_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Prepare CSV file
csv_filename = 'tackle_analysis_R1_1.csv'
csv_header = ['frame', 'ball_carrier_speed', 'tackler_speed', 'impact_detected', 'head_below_wrists', 'head_back_hips_inline']

ball_carrier_positions = []
tackler_positions = []
frame_window = 5
impact_detected = False
t_impact = None

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        if len(results[0].keypoints) >= 2:  # Ensure we have detected at least two people
            # Sort keypoints by x-coordinate to ensure left (ball carrier) and right (tackler) are correct
            sorted_keypoints = sorted(results[0].keypoints, key=lambda x: x.xy[0, 0, 0])
            ball_carrier = sorted_keypoints[0].xy[0].cpu().numpy()
            tackler = sorted_keypoints[1].xy[0].cpu().numpy()

            ball_carrier_positions.append(ball_carrier[0])  # Using hip point (index 0)
            tackler_positions.append(tackler[0])
            
            if len(ball_carrier_positions) > frame_window:
                ball_carrier_positions.pop(0)
                tackler_positions.pop(0)

            # Calculate speeds
            ball_carrier_speed = 0
            tackler_speed = 0
            if len(ball_carrier_positions) > 1:
                ball_carrier_speed = calculate_speed(ball_carrier_positions[-2], ball_carrier_positions[-1], fps)
                tackler_speed = calculate_speed(tackler_positions[-2], tackler_positions[-1], fps)

            # Detect impact
            if not impact_detected:
                impact_detected = detect_impact(np.array(ball_carrier_positions), np.array(tackler_positions))
                if impact_detected:
                    t_impact = frame_count / fps

            # Analyze tackle
            head_below_wrists, head_back_hips_inline = analyze_tackle(ball_carrier, tackler)

            # Write data to CSV
            csv_writer.writerow([frame_count, ball_carrier_speed, tackler_speed, impact_detected, head_below_wrists, head_back_hips_inline])

            # Draw labels and indicators
            draw_labels_and_indicators(annotated_frame, ball_carrier, tackler, head_below_wrists)

            # Annotate frame with analysis results
            cv2.putText(annotated_frame, f"Ball Carrier Speed: {ball_carrier_speed:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Tackler Speed: {tackler_speed:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if impact_detected:
                cv2.putText(annotated_frame, f"Impact Detected at t={t_impact:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Head Below Wrists: {head_below_wrists}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Head-Back-Hips Inline: {head_back_hips_inline}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(annotated_frame)
        cv2.imshow("Rugby Tackle Analysis", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

logging.info(f"Analysis complete. Results saved to {csv_filename} and {output_path}")