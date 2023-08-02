dataset_info = dict(
    dataset_name='leipzigchimp',
    paper_info=dict(
        author='',
        title='',
        container='',
        year='',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='pelvis', id=0, color=[51, 153, 255], type='lower', swap=''),
        1:
        dict(name='right_knee', id=1, color=[255, 128, 0], type='lower', swap='left_knee'),
        2:
        dict(name='right_ankle', id=2, color=[255, 128, 0], type='lower', swap='left_ankle'),
        3:
        dict(name='left_knee', id=3, color=[0, 255, 0], type='lower', swap='right_knee'),
        4:
        dict(name='left_ankle', id=4, color=[0, 255, 0], type='lower', swap='right_ankle'),
        5:
        dict(name='neck', id=5, color=[51, 153, 255], type='upper', swap=''),
        6:
        dict(name='upper_lip', id=6, color=[51, 153, 255], type='upper', swap=''),
        7:
        dict(name='lower_lip', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(name='right_eye', id=8, color=[255, 128, 0], type='upper', swap='left_eye'),
        9:
        dict(name='left_eye', id=9, color=[0, 255, 0], type='upper', swap='right_eye'),
        10:
        dict(name='right_shoulder', id=10, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        11:
        dict(name='right_elbow', id=11, color=[255, 128, 0], type='upper', swap='left_elbow'),
        12:
        dict(name='right_wrist', id=12, color=[255, 128, 0], type='upper', swap='left_wrist'),
        13:
        dict(name='left_shoulder', id=13, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        14:
        dict(name='left_elbow', id=14, color=[0, 255, 0], type='upper', swap='right_elbow'),
        15:
        dict(name='left_wrist', id=15, color=[0, 255, 0], type='upper', swap='right_wrist')
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('right_knee', 'pelvis'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('pelvis', 'left_knee'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('left_knee', 'left_ankle'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('pelvis', 'neck'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('neck', 'lower_lip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('lower_lip', 'upper_lip'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('upper_lip', 'right_eye'), id=7, color=[255, 128, 0]),
        8:
        dict(link=('upper_lip', 'left_eye'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('neck', 'right_shoulder'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('right_shoulder', 'right_elbow'), id=10, color=[255, 128, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('neck', 'left_shoulder'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('left_shoulder', 'left_elbow'), id=13, color=[0, 255, 0]),
        14:
        dict(link=('left_elbow', 'left_wrist'), id=14, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    # Adapted from COCO dataset.
    sigmas=[
        0.026, 0.083, 0.089, 0.083, 0.089, 0.026, 0.026, 0.026, 0.026, 0.026,
        0.179, 0.072, 0.062, 0.179, 0.072, 0.062
    ])
