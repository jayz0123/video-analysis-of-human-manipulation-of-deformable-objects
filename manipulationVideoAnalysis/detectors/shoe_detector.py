


from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/mmdet_configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '../resources/test_frames/rgb/frame000153.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')