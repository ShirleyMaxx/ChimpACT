import numpy as np
import os
import os.path as osp
import pickle
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize pose estimation model results.')
    parser.add_argument(
        '--input', help='model name of saved pose estimation results', default='td-hm_cpm')
    parser.add_argument(
        '--output', help='path to save visualization results', default='work_dirs/vis_results')
    parser.add_argument(
        '--vid-name', help='video name to be visualized', default='Azibo_ObsChimp_2016_08_02_c_clip_33000_34000')
    parser.add_argument(
        '--interval', type=int, help='sampling interval of results to be visualizaed', default=10)
    parser.add_argument(
        '--gt', help='visualize GT annotations', action='store_true')
    parser.add_argument(
        '--vis-kpt', help='visualize body keypoints', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    use_gt = args.gt
    model_name = args.input
    assert model_name in ['td-reg_res50', 'td-reg_res101', 'td-reg_res152', 'td-reg_mobilenetv2_rle', 'td-reg_res50_rle', 'td-reg_res101_rle', 'td-reg_res152_rle', 'td-hm_cpm', 'td-hm_hourglass52', 'td-hm_mobilenetv2', 'td-hm_res50', 'td-hm_res101', 'td-hm_res152', 'td-hm_hrnet-w32', 'td-hm_hrnet-w48', 'td-hm_res50_dark', 'td-hm_res101_dark', 'td-hm_res152_dark', 'td-hm_hrnet-w32_dark', 'td-hm_hrnet-w48_dark', 'td-hm_hrformer-small', 'td-hm_hrformer-base']
    path = f'work_dirs/{model_name}.pkl'
    save_dir = osp.join(args.output, f'vis_pred_pose_{model_name}' if not use_gt else 'vis_gt_pose')
    os.makedirs(save_dir, exist_ok=True)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    img_pose_dict = defaultdict(list)

    for datum in data:
        if use_gt:
            img_pose_dict[datum['img_path']].append(np.array(datum['raw_ann_info']['keypoints']).reshape(-1, 3))  
        else:
            img_pose_dict[datum['img_path']].append(np.array(datum['pred_instances']['keypoints']).reshape(-1, 2))  

    joint_names = ['root', 'rknee', 'rankle', 'lknee', 'lankle', 'neck', 'ulip', 'llip', 'reye', 'leye', 'rshoul', 'relbow', 'rwrist', 'lshoul', 'lelbow', 'lwrist']
    LEFT_SKELE = [(0, 3), (3, 4), (5, 13), (13, 14), (14, 15), (6, 9)]
    RIGHT_SKELE = [(0, 1), (1, 2), (5, 10), (10, 11), (11, 12), (6, 8)]
    skeleton = [(0,1), (1,2), (0,3), (3,4), (0,5), (6,7), (6,8), (6,9), (5,10), (10,11), (11,12), (5,13), (13,14), (14,15)]
    left_keypoint = [3,4,9,13,14,15]
    right_keypoint = [1,2,8,10,11,12]
    mid_keypoint = [0,5,6,7]

    left_color = np.array((0, 240, 0))/255
    right_color = np.array((250, 126, 0))/255
    mid_color = np.array((52, 143, 235))/255

    counter = 0
    vid_name = args.vid_name
    for img_path in tqdm(img_pose_dict.keys(), dynamic_ncols=True):
        counter += 1
        img_name = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        if img_path.split('/')[-2] != vid_name:
            continue
        if counter % int(args.interval):
            continue
        img = cv2.imread(img_path)
        multiposes = img_pose_dict[img_path]
        plt.close('all')
        size = 10
        fig = plt.figure(figsize=(1*size, 1*size))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img[...,::-1])
        for poses in multiposes:
            vis = np.ones_like(poses[:,:1])
            if use_gt:
                vis = poses[:,2]
                poses = poses[:,:2]
            for i, jt in enumerate(skeleton):
                if jt in LEFT_SKELE:
                    color = left_color
                elif jt in RIGHT_SKELE:
                    color = right_color
                else:
                    color = mid_color
                xs, ys = [np.array([poses[jt[0], j], poses[jt[1], j]]) for j in range(2)]
                xvis, yvis = vis[jt[0]], vis[jt[1]]
                if xvis == 0 or yvis == 0:
                    continue
                ax.plot(xs, ys, lw=2, ls='-', c=color, solid_capstyle='round', zorder=1)
            # draw keypoints
            if args.vis_kpt:
                ax.scatter(poses[left_keypoint, 0], poses[left_keypoint, 1], s=20, c=left_color, zorder=2)
                ax.scatter(poses[right_keypoint, 0], poses[right_keypoint, 1], s=20, c=right_color, zorder=2)
                ax.scatter(poses[mid_keypoint, 0], poses[mid_keypoint, 1], s=20, c=mid_color, zorder=2)
        plt.axis("off")
        save_path = osp.join(save_dir, img_name.replace('.jpg', '.png'))
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=-0.1)


if __name__ == '__main__':
    main()