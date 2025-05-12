#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 18:54:07 2024

@author: Rowan
"""

import cv2

def main():
    # Open the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()