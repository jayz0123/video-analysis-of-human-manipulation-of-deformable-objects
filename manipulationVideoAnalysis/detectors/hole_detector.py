import sys
import numpy as np
import cv2 as cv


class HoleDetector():
    def __init__(self):
        detector = cv.SimpleBlobDetector_create()
        params = cv.SimpleBlobDetector_Params()

        params.minThreshold = 80
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 65
        params.maxArea = 250

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 1

        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.maxConvexity = 1

        params.minDistBetweenBlobs = 0

        # Create a detector with the parameters
        self.detector = cv.SimpleBlobDetector_create(params)

    def detect(self, frame):
        blurred = cv.bilateralFilter(frame, 9, 70, 70)
        keypoints = self.detector.detect(blurred)

        return keypoints
