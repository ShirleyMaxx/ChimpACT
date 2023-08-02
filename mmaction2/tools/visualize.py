import numpy as np
import os
import os.path as osp
import pickle
import pandas
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import argparse
import torch

BEHAVE_ID_SET = {
    0: 'moving',
    1: 'climbing',
    2: 'resting',
    3: 'sleeping',
    4: 'sol. obj. playing',
    5: 'eating',
    6: 'mani. obj.',
    7: 'grooming',
    8: 'being groomed',
    9: 'aggressing',
    10: 'embracing',
    11: 'begging',
    12: 'being begged',
    13: 'taking obj.',
    14: 'losing obj.',
    15: 'carring',
    16: 'being carried',
    17: 'nursing',
    18: 'being nursed',
    19: 'playing',
    20: 'touching',
    21: 'erection',
    22: 'displaying',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize action detection model results.')
    parser.add_argument(
        '--input', help='model name of saved action detection results', default='slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb')
    parser.add_argument(
        '--output', help='path to save visualization results', default='work_dirs/vis_results')
    parser.add_argument(
        '--vid-name', help='video name to be visualized', default='Azibo_ObsChimp_2017_06_22_c_clip_44000_45000')
    parser.add_argument(
        '--interval', type=int, help='sampling interval of results to be visualizaed', default=10)
    parser.add_argument(
        '--conf-thres', type=float, help='confidence threshold', default=0.5)
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.input
    conf_thres = float(args.conf_thres)
    vis_vid_name = args.vid_name
    interval = int(args.interval)
    assert model_name in ['slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_chimp-rgb','slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb','slowfast_kinetics400-pretrained-r50-context_8xb8-8x8x1-20e_chimp-rgb','slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb','slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb','slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-10e_chimp-rgb','slowonly_k400-pre-r50_8xb8-8x8x1-20e_chimp-rgb','slowonly_k400-pre-r50-context_8xb8-8x8x1-20e_chimp-rgb','slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb','slowonly_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb','slowonly-lfb-avg_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb','slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb','slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb']
    path = f'work_dirs/result_{model_name}.pkl'
    save_dir = osp.join(args.output, f'vis_pred_action_{model_name}')
    save_img_dir = osp.join(args.output, f'vis_pred_action_{model_name}', 'imgs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    img_action_dict = defaultdict(list)

    for datum in data:
        vidname = datum['video_id']
        imgname = datum['timestamp']
        img_path = osp.join('data/ChimpACT_processed', vidname, f'{imgname:06d}.jpg')

        img_path_idx = vidname + '_' + f'{imgname:06d}.jpg'

        for bbox, conf_arr in zip(datum['pred_instances']['bboxes'], datum['pred_instances']['scores']):
            pred_actions = torch.nonzero(conf_arr[1:]>conf_thres).squeeze(0)   # the first class is 'null'
            for pred in pred_actions:
                img_action_dict[img_path_idx].append([img_path, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), int(pred)])

    counter = 0
    color_box = (84, 130, 53)
    color_text = (0, 0, 0)
    color_bg = (233, 235, 235)
    txt_w_min, txt_h_min = 10, 10
    save_path = osp.join(save_dir, vis_vid_name+'.mp4')
    videowriter = imageio.get_writer(save_path, fps=25)

    for img_path_idx in tqdm(img_action_dict.keys(), dynamic_ncols=True):
        counter += 1
        vidname = img_action_dict[img_path_idx][0][0].split('/')[-2]
        if vidname != vis_vid_name:
            continue
        img_path = img_action_dict[img_path_idx][0][0]

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        raw_preds = img_action_dict[img_path_idx]
        multiboxes = defaultdict(list)
        for items in raw_preds:
            assert items[0] == img_path
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(float(items[1]*img_w)), int(float(items[2]*img_h)), int(float(items[3]*img_w)), int(float(items[4]*img_h))
            multiboxes[(bbox_x1, bbox_y1, bbox_x2, bbox_y2)].append(int(items[-1]))

        for box_info in multiboxes.keys():
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = box_info
            cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color_box, 1)
            behaviors = ''
            for behave_idx in multiboxes[box_info]:
                behaviors += f'{BEHAVE_ID_SET[behave_idx]}, '
            txt_attrs = f"{behaviors[:-2]}"
            txt_size = cv2.getTextSize(f"{txt_attrs}", 0, 0.5, 1)[0]
            txt_w, txt_h = txt_size[:2]
            txt_w_max, txt_h_max = img_w - txt_w, img_h - txt_h
            txt_x, txt_y = min(max(bbox_x1 -10, txt_w_min), txt_w_max), min(max(bbox_y1 - 10, txt_h_min), txt_h_max)
            cv2.rectangle(img, (txt_x+5, txt_y-30), (txt_x+txt_w+160, txt_y+txt_h), color_bg, thickness=-1)
            cv2.putText(img, f"{txt_attrs}", (txt_x+10, txt_y), cv2.FONT_HERSHEY_DUPLEX, 1, color_text, 2)

        if not counter % interval:
            save_img_path = osp.join(save_img_dir, img_path_idx.split('/')[-1].replace('.jpg', '.png'))
            cv2.imwrite(save_img_path, img)
        videowriter.append_data(img[...,::-1])
    
    videowriter.close()


if __name__ == '__main__':
    main()