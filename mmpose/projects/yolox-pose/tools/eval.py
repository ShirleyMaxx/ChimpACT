import numpy as np
import os
import os.path as osp
import pickle

from mmpose.evaluation.functional import pose_pck_accuracy, keypoint_pck_accuracy

path_dir = '/ceph/home/yixin01/xiaoxuan/primates/ChimpAI/mmpose/results_my'

def get_path(path_dir):
    return os.listdir(path_dir)

def evaluate_pckh(pred, gt, gt_vis=None, gt_bbox=None, threshold=0.05):

    # pred, gt: (N, K, 2)
    # gt_vis: (N, K)
    distance = np.sqrt(np.sum((gt - pred)**2, axis=2))  # (N, K)
    # import ipdb; ipdb.set_trace()
    if gt_bbox is None:
        headsize = np.sqrt(np.sum(((0.5*(gt_pose[:, 10] + gt_pose[:, 12])) - gt_pose[:, 7])**2, axis=1))*2   # (N,)
        thres = headsize[:, None] * threshold   # (N, 1)
    else:
        thres = gt_bbox[:, None]*threshold

    if gt_vis is None:
        joint_detection_rate = (distance <= thres).sum(-1) / gt_pose.shape[1]
    else:
        # 0 for invisible joints, and 1 for visible. Invisible joints will be ignored for accuracy calculation.
        distance_valid = gt_vis != 0    # (N, K)
        joint_detection_rate = ((distance <= thres)*distance_valid).sum(-1) / distance_valid.sum(-1)
    pckh = np.mean(joint_detection_rate)

    return pckh


path_list = get_path(path_dir)

for file in path_list:
    # if 'td-reg_res' not in file:
    #     continue
    with open(osp.join(path_dir, file), 'rb') as f:
        data = pickle.load(f)

    pred_pose = []
    gt_pose = []
    gt_vis = []
    gt_bbox = []

    for datum in data:
        # import ipdb; ipdb.set_trace()
        gt_pose.append(np.array(datum['raw_ann_info']['keypoints']).reshape(-1, 3)[:, :2])
        gt_vis.append(np.array(datum['raw_ann_info']['keypoints']).reshape(-1, 3)[:, 2])
        pred_pose.append(np.array(datum['pred_instances']['keypoints']).reshape(-1, 2))
        gt_bbox.append(max(datum['raw_ann_info']['bbox'][2], datum['raw_ann_info']['bbox'][3]))

    gt_bbox = np.array(gt_bbox)
    gt_pose = np.array(gt_pose)
    gt_vis = np.array(gt_vis).astype(np.int32)
    gt_vis = np.minimum(1, gt_vis)

    pred_pose = np.array(pred_pose)
    # acc, avg_acc, cnt = keypoint_pck_accuracy(pred_pose, gt_pose, gt_vis, thr=0.5, norm_factor=np.ones((int(pred_pose.shape[0]), 2), dtype=np.float32))

    pckh0_05 = evaluate_pckh(pred_pose, gt_pose, gt_vis, gt_bbox, 0.05)
    pckh0_1 = evaluate_pckh(pred_pose, gt_pose, gt_vis, gt_bbox, 0.1)
    print(f'{file} |||||  pckh@0.05:  {pckh0_05*100:.1f} , pckh@0.1:  {pckh0_1*100:.1f} ')