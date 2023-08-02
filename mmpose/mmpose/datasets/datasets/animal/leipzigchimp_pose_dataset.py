# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import exists, get_local_path

from mmpose.registry import DATASETS
from xtcocotools.coco import COCO
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class LeipzigChimpDataset(BaseCocoStyleDataset):
    """LeipzigChimp Dataset for pose estimation.

    LeipzigChimp keypoints::

        0: 'pelvis',
        1: 'right_knee',
        2: 'right_ankle'
        3: 'left_knee',
        4: 'left_ankle',
        5: 'neck',
        6: 'upper_lip',
        7: 'lower_lip',
        8: 'right_eye',
        9: 'left_eye',
        10: 'right_shoulder',
        11: 'right_elbow',
        12: 'right_wrist',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/leipzigchimp.py')

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""
        print(self.ann_file)
        assert exists(self.ann_file), 'Annotation file does not exist'
        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)

        # [{'id': 1, 'name': 'chimpanzee'}]
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):
                # remove null annotations
                if not (ann['iskeyframe'] and len(ann['keypoints'])):
                    continue
                
                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img))

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_list.append(instance_info)

        return instance_list, image_list

