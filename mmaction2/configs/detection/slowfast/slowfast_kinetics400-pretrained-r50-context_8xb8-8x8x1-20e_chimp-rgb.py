_base_ = ['slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_chimp-rgb.py']

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(with_global=True),
        bbox_head=dict(in_channels=4608)))