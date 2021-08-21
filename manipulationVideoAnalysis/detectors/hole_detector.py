import sys
import numpy as np
import cv2 as cv


class HoleDetector():
    def __init__(self):
        detector = cv.SimpleBlobDetector_create()
        params = cv.SimpleBlobDetector_Params()

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 500

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0

        # Create a detector with the parameters
        self.detector = cv.SimpleBlobDetector_create(params)

    def detect_keypoints(self, frame):
        keypoints = self.detector.detect(frame)

        return keypoints
