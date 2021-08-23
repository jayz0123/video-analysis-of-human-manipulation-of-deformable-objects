import cv2 as cv
import numpy as np


class AgletDetector():
    def __init__(self):
        self.lower = np.array([165, 150, 100])
        self.upper = np.array([179, 255, 255])

    def detect(self, frame):
        blurred = cv.bilateralFilter(frame, 40, 25, 25)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, self.lower, self.upper)

        mask = cv.dilate(mask, None, iterations=2)

        contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        center = None

        if not contours:
            return []

        contour = max(contours, key=cv.contourArea)
        rect = cv.minAreaRect(contour)
        ((x, y), (width, height), rotation) = rect

        bbox = cv.boxPoints(rect)
        bbox = np.int64(bbox)

        return bbox
