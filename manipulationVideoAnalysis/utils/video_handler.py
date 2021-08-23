import numpy as np
import cv2 as cv
import glob
from itertools import zip_longest
from mmpose.apis import vis_pose_result
from detectors.hole_detector import HoleDetector
from detectors.hand_detector import HandDetector
from detectors.aglet_detector import AgletDetector


class VideoHandler():
    def __init__(self,
                 rgb_path: str,
                 depth_path: str,
                 hole_detecting=False,
                 hand_detecting=False,
                 aglet_detecting=False):
        self.rgb_frames = []
        self.depth_frames = []

        self.hole_detecting = hole_detecting
        self.hand_detecting = hand_detecting
        self.aglet_detecting = aglet_detecting

        self.hole_keypoints_all_frames = []
        self.hand_bboxes_all_frames = []
        self.aglet_bbox_all_frames = []

        for filename in sorted(glob.glob(rgb_path + 'frame*.png')):
            rgb_frame = cv.imread(filename, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            self.rgb_frames.append(rgb_frame)

        for filename in sorted(glob.glob(depth_path + 'frame*.png')):
            depth_frame = cv.imread(filename, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            self.depth_frames.append(depth_frame)

        if self.hole_detecting:
            hole_detector = HoleDetector()

            for frame in self.rgb_frames:
                self.hole_keypoints_all_frames.append(hole_detector.detect(frame))

        if self.hand_detecting:
            self.hand_detector = HandDetector()

            for frame in self.rgb_frames:
                self.hand_bboxes_all_frames.append(self.hand_detector.detect(frame))

        if self.aglet_detecting:
            self.aglet_detector = AgletDetector()

            for frame in self.rgb_frames:
                self.aglet_bbox_all_frames.append(self.aglet_detector.detect(frame))

    def _visualise_depth_frame(self, frame):
        normed = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        colorised = cv.applyColorMap(normed, cv.COLORMAP_BONE)

        return colorised

    def _render_hole_keypoints(self, frame, keypoints):
        overlay = frame.copy()

        for k in keypoints:
            cv.circle(overlay, (int(k.pt[0]), int(k.pt[1])), int(k.size / 2), (0, 0, 255), -1)
            cv.line(overlay, (int(k.pt[0]) - 20, int(k.pt[1])), (int(k.pt[0]) + 20, int(k.pt[1])), (0, 0, 0), 3)
            cv.line(overlay, (int(k.pt[0]), int(k.pt[1]) - 20), (int(k.pt[0]), int(k.pt[1]) + 20), (0, 0, 0), 3)

        opacity = 0.5
        rendered_frame = cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

        return rendered_frame

    def _render_hand_bboxes(self, frame, bboxes):
        rendered_frame = vis_pose_result(
            self.hand_detector.pose_model,
            frame,
            bboxes,
            dataset=self.hand_detector.dataset,
            kpt_score_thr=self.hand_detector.kpt_threshold,
            radius=self.hand_detector.kpt_radius,
            thickness=self.hand_detector.thickness,
            show=False)

        return rendered_frame

    def _render_aglet_bbox(self, frame, bbox):
        if len(bbox) == 0:
            return frame

        overlay = frame.copy()
        cv.drawContours(overlay, [bbox], 0, (0, 255, 255), 2)

        rendered_frame = overlay

        return rendered_frame

    def replay(self, mode: str):
        # play the rgb video
        if mode == 'rgb':
            for frame, hole_keypoints, hand_bboxes, aglet_bbox in zip_longest(
                    self.rgb_frames,
                    self.hole_keypoints_all_frames,
                    self.hand_bboxes_all_frames,
                    self.aglet_bbox_all_frames):
                if self.hole_detecting:
                    frame = self._render_hole_keypoints(frame, hole_keypoints)

                if self.hand_detecting:
                    frame = self._render_hand_bboxes(frame, hand_bboxes)

                if self.aglet_detecting:
                    frame = self._render_aglet_bbox(frame, aglet_bbox)

                cv.imshow('RGB', frame)

                if cv.waitKey(100) == ord('q'):
                    break

        # play the depth video
        elif mode == 'depth':
            for frame, hole_keypoints in zip_longest(self.depth_frames, self.hole_keypoints_all_frames):
                if self.hole_detecting:
                    frame= self._render_hole_keypoints(frame, hole_keypoints)

                cv.imshow('Depth', self._visualise_depth_frame(frame))

                if cv.waitKey(100) == ord('q'):
                    break

        # play the rbg and depth video together
        elif mode == 'both':
            if len(self.rgb_frames) != len(self.depth_frames):
                raise Exception('Video frames are inconsistent!')

            for rgb_frame, depth_frame, hole_keypoints in zip_longest(self.rgb_frames, self.depth_frames, self.hole_keypoints_all_frames):
                if self.hole_detecting:
                    rgb_frame = self._render_hole_keypoints(rgb_frame, hole_keypoints)
                    depth_frame = self._render_hole_keypoints(depth_frame, hole_keypoints)

                depth_frame = self._visualise_depth_frame(depth_frame)
                frame_concatenated = np.concatenate((rgb_frame, depth_frame), axis=1)

                cv.imshow('RGB and Depth', frame_concatenated)

                if cv.waitKey(100) == ord('q'):
                    break

        else:
            raise Exception('Undefined mode!')

        cv.destroyAllWindows()