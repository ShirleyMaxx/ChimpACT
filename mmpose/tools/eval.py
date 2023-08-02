import numpy as np
import os
import os.path as osp
import pickle
from pandas import DataFrame

path_dir = 'work_dirs'

def get_path(path_dir):
    path_list = [osp.join(path_dir, path) for path in os.listdir(path_dir) if '.pkl' in path]
    return path_list

def evaluate_pck(pred, gt, gt_vis=None, gt_bbox=None, threshold=0.05):

    # pred, gt: (N, K, 2)
    # gt_vis: (N, K)
    distance = np.sqrt(np.sum((gt - pred)**2, axis=2))  # (N, K)
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
        per_joint_pck = ((distance <= thres)*distance_valid).sum(0) / distance_valid.sum(0)   # (J,)
    pck = np.mean(joint_detection_rate)

    return pck, per_joint_pck

joint_names = ['root', 'rknee', 'rankle', 'lknee', 'lankle', 'neck', 'ulip', 'llip', 'reye', 'leye', 'rshoul', 'relbow', 'rwrist', 'lshoul', 'lelbow', 'lwrist']
per_joint_pck0_1_file = {'name': [], 'root': [], 'rknee': [], 'rankle': [], 'lknee': [], 'lankle': [], 'neck': [], 'ulip': [], 'llip': [], 'reye': [], 'leye': [], 'rshoul': [], 'relbow': [], 'rwrist': [], 'lshoul': [], 'lelbow': [], 'lwrist': []}

path_list = get_path(path_dir)

for file in path_list:
    with open(file, 'rb') as f:
        data = pickle.load(f)

    pred_pose = []
    gt_pose = []
    gt_vis = []
    gt_bbox = []

    for datum in data:
        gt_pose.append(np.array(datum['raw_ann_info']['keypoints']).reshape(-1, 3)[:, :2])
        gt_vis.append(np.array(datum['raw_ann_info']['keypoints']).reshape(-1, 3)[:, 2])
        pred_pose.append(np.array(datum['pred_instances']['keypoints']).reshape(-1, 2))
        gt_bbox.append(max(datum['raw_ann_info']['bbox'][2], datum['raw_ann_info']['bbox'][3]))

    gt_bbox = np.array(gt_bbox)
    gt_pose = np.array(gt_pose)
    gt_vis = np.array(gt_vis).astype(np.int32)
    gt_vis = np.minimum(1, gt_vis)

    pred_pose = np.array(pred_pose)

    pck0_05, per_joint_pck0_05 = evaluate_pck(pred_pose, gt_pose, gt_vis, gt_bbox, 0.05)
    pck0_1, per_joint_pck0_1 = evaluate_pck(pred_pose, gt_pose, gt_vis, gt_bbox, 0.1)
    print(f'{file.split("/")[-1][:-4]} ===>  PCK@0.05:  {pck0_05*100:.1f} , PCK@0.1:  {pck0_1*100:.1f} ')

    per_joint_pck0_1_file['name'].append(file)
    for jname, jpck in zip(joint_names, per_joint_pck0_1):
        per_joint_pck0_1_file[jname].append(jpck)

df = DataFrame(per_joint_pck0_1_file)
df.to_excel('work_dirs/results_perjoint_pck0_1.xlsx')