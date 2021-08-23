import cv2 as cv
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmdet.apis import inference_detector, init_detector


class HandDetector():
    def __init__(self):
        self.det_config = 'detectors/configs/mmdet_configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1class.py'
        self.det_checkpoint = 'detectors/checkpoints/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'
        self.pose_config = 'detectors/configs/mmpose_configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'
        self.pose_checkpoint = 'detectors/checkpoints/res50_onehand10k_256x256-e67998f6_20200813.pth'
        # self.det_config = 'detectors/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py'
        # self.det_checkpoint = 'detectors/checkpoints/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
        # self.pose_config = 'detectors/configs/hand/2d_kpt_sview_rgb_img/deeppose/onehand10k/res50_onehand10k_256x256.py'
        # self.pose_checkpoint = 'detectors/checkpoints/deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth'

        self.device = 'cuda:0'
        self.det_cat_id = 1
        self.bbox_threshold = 0.5
        self.kpt_threshold = 0.5
        self.kpt_radius = 4
        self.thickness = 1

        # build the detection model and the pose model
        self.det_model = init_detector(self.det_config, self.det_checkpoint, device=self.device.lower())
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint, device=self.device.lower())
        self.dataset = self.pose_model.cfg.data['test']['type']

    def _process_det_results(self, det_results, cat_id=1):
        if isinstance(det_results, tuple):
            det_result = det_results[0]
        else:
            det_result = det_results

        bboxes = det_result[cat_id - 1]
        person_results = []
        for bbox in bboxes:
            person = {'bbox': bbox}
            person_results.append(person)

        return person_results

    def detect(self, frame):
        det_results = inference_detector(self.det_model, frame)
        person_results = self._process_det_results(det_results, self.det_cat_id)

        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            frame,
            person_results,
            bbox_thr=self.bbox_threshold,
            format='xyxy',
            dataset=self.dataset,
            return_heatmap=False,
            outputs=None)

        return pose_results
