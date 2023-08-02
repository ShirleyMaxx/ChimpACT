_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/leipzigchimp.py', '../../_base_/default_runtime.py'
]

img_scale = (800, 1440)
samples_per_gpu = 2

model = dict(
    type='OCSORT',
    detector=dict(
        backbone=dict(
            norm_cfg=dict(requires_grad=False),
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                bbox_coder=dict(clip_border=False),
                num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='OCSORTTracker',
        obj_score_thr=0.7,
        init_track_thr=0.8,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.001 / 8 * samples_per_gpu,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)

# some hyper parameters
total_epochs = 5
num_last_epochs = 5
resume_from = None
interval = 5

# learning policy
lr_config = dict(policy='step', step=[3])



checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))