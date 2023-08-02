# This script converts LeipzigChimp COCO-format labels into AVA style.
#
# NOTE! Action label should start at 1  
#
#     {subset}_action_excluded_timestamps.csv
#     {subset}_action_gt.pkl
#     {subset}_action.csv
# 
# ##############################################


import json
import numpy as np
import argparse
import os
import os.path as osp
import sys
from tqdm import tqdm
from pathlib import Path
import glob
import cv2
import imageio
import shutil
from pycocotools.coco import COCO
from collections import defaultdict
from pandas import DataFrame
from termcolor import colored
import pickle
import sys
from mmtrack.datasets.parsers import CocoVID

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ChimpACT COCO-VID format annotation to AVA labels.')
    parser.add_argument(
        '-i', '--input', help='path of COCO-VID formatted ChimpACT data', default='data/ChimpACT_release')
    parser.add_argument(
        '-o', '--output', help='path of COCO-VID formatted ChimpACT data', default='data/ChimpACT_processed')
    return parser.parse_args()

def main():
    args = parse_args()
    ori_dir = args.input
    data_dir = args.output
    os.makedirs(osp.join(data_dir, 'annotations', 'action'), exist_ok=True)
    shutil.copy(osp.join(ori_dir, 'action_list.txt'), osp.join(data_dir, 'annotations', 'action', 'action_list.txt'))
    for subset in ['train', 'val', 'test']:
        annot_file = CocoVID(osp.join(data_dir, 'annotations', f'{subset}.json'))
        annot_file_action_path = osp.join(data_dir, 'annotations', 'action', f'{subset}_action.csv')
        annot_file_action_excluded_timestamps_path = osp.join(data_dir, 'annotations', 'action', f'{subset}_action_excluded_timestamps.csv')
        annot_file_action_proposal_gt_path = osp.join(data_dir, 'annotations', 'action', f'{subset}_action_gt.pkl')

        annot_file_action, annot_file_action_excluded_timestamps = [], []
        annot_file_action_proposal_gt = dict()
        vid_ids = annot_file.get_vid_ids()
        cat_ids = annot_file.get_cat_ids()
        for vid_id in vid_ids:
            img_ids = annot_file.get_img_ids_from_vid(vid_id)
            for img_id in tqdm(img_ids):
                img_info = annot_file.load_imgs([img_id])[0]
                img_w, img_h = img_info['width'], img_info['height']
                img_path = osp.join(data_dir, subset, 'images', img_info['file_name'])
                img_path_rel = img_path[len(data_dir)+1:-11]

                frame_idx = img_info['frame_id']

                ann_ids = annot_file.get_ann_ids(img_ids=[img_id], cat_ids=cat_ids)
                objs = annot_file.load_anns(ann_ids)

                if len(objs) == 0:      # no annotation frame
                    annot_file_action_excluded_timestamps.append(f'{img_path_rel},{frame_idx}\n')
                    continue
                bbox_list = []
                for obj in objs:
                    # video_identifier, time_stamp, lt_x, lt_y, rb_x, rb_y, label, entity_id
                    bbox_id = obj['instance_id']
                    behave_list = obj['behaviors']
                    bbox_x1, bbox_y1, bbox_w, bbox_h = obj['bbox']
                    bbox_x2, bbox_y2 = bbox_x1 + bbox_w, bbox_y1 + bbox_h

                    for behave in behave_list:
                        annot_file_action.append(f'{img_path_rel},{frame_idx},{bbox_x1/img_w:.3f},{bbox_y1/img_h:.3f},{bbox_x2/img_w:.3f},{bbox_y2/img_h:.3f},{behave+1},{bbox_id}\n')

                    # lt_x, lt_y, rb_x, rb_y, score=1.0
                    bbox_list.append([bbox_x1/img_w, bbox_y1/img_h, bbox_x2/img_w, bbox_y2/img_h, 1.0])
                # {'video_identifier,time_stamp': np.array((N,5))}
                annot_file_action_proposal_gt[f'{img_path_rel},{frame_idx:04d}'] = np.array(bbox_list)


        with open(annot_file_action_path, 'w') as f:
            for line in annot_file_action:
                f.write(line)
        
        with open(annot_file_action_excluded_timestamps_path, 'w') as f:
            for line in annot_file_action_excluded_timestamps:
                f.write(line)

        with open(annot_file_action_proposal_gt_path, 'wb') as f:
            pickle.dump(annot_file_action_proposal_gt, f)

if __name__ == "__main__":
    main()
