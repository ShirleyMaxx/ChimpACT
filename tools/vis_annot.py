import json
import numpy as np
import argparse
import os
import os.path as osp
from pathlib import Path
from collections import defaultdict
import imageio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
np.set_printoptions(suppress=True)
from mmtrack.datasets.parsers import CocoVID

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

BEHAVE_SOCIAL = {
    'grooming': 0,
    'being groomed': 1,
    'aggressing': 2,
    'embracing': 3,
    'begging': 4,
    'be begging from': 5,
    'taking obj.': 6,
    'losing obj.': 7,
    'carrying': 8,
    'being carryied': 9,
    'nursing': 10,
    'being nursed': 11,
    'playing': 12,
    'touching': 13,
}

NAME_SET_REV = {
    'Azibo': 0,
    'Bambari': 1,
    'Bangolo': 2,
    'Corrie': 3,
    'Dorien': 4,
    'Fraukje': 5,
    'Frodo': 6,
    'Kara': 7,
    'Kisha': 8,
    'Kofi': 9,
    'Lobo': 10,
    'Lome': 11,
    'Maja': 12,
    'Makeni': 13,
    'Natascha': 14,
    'Ohini': 15,
    'Riet': 16,
    'Robert': 17,
    'Sandra': 18,
    'Swela': 19,
    'Taï': 20,
    'Ulla': 21,
    'Youma': 22,
    '': 23
}

NAME_SET = {
    0: ['Azibo', 'Male', '2015-4-14'],
    1: ['Bambari', 'Famale', '2000-12-8'],
    2: ['Bangolo', 'Male', '2009-7-5'],
    3: ['Corrie', 'Famale', '1976-12-12'],
    4: ['Dorien', 'Famale', '1980-10-22'],
    5: ['Fraukje', 'Famale', '1976-4-6'],
    6: ['Frodo', 'Male', '1993-11-28'],
    7: ['Kara', 'Famale', '2005-6-23'],
    8: ['Kisha', 'Famale', '2004-3-4'],
    9: ['Kofi', 'Male', '2005-7-7'],
    10: ['Lobo', 'Male', '2004-4-21'],
    11: ['Lome', 'Male', '2001-8-11'],
    12: ['Maja', 'Famale', '1986-5-1'],
    13: ['Makeni', 'Male', '2018-3-14'],
    14: ['Natascha', 'Famale', '1980-3-28'],
    15: ['Ohini', 'Male', '2016-3-25'],
    16: ['Riet', 'Famale', '1977-11-11'],
    17: ['Robert', 'Male', '1975-12-1'],
    18: ['Sandra', 'Famale', '1993-6-9'],
    19: ['Swela', 'Famale', '1995-10-19'],
    20: ['Taï', 'Famale', '2002-8-12'],
    21: ['Ulla', 'Famale', '1977-6-8'],
    22: ['Youma', 'Famale', '2018-3-25'],
    23: ['', '', '']
}

locomotion_ids = [0,1,2,3]
object_ids = [4,5,6]
social_ids = [7,8,9,10,11,12,13,14,15,16,17,18,19,20]
others_ids = [21,22]

LEFT_SKELE = [(0, 3), (3, 4), (5, 13), (13, 14), (14, 15), (6, 9)]
RIGHT_SKELE = [(0, 1), (1, 2), (5, 10), (10, 11), (11, 12), (6, 8)]
skeleton = [(0,1), (1,2), (0,3), (3,4), (0,5), (6,7), (6,8), (6,9), (5,10), (10,11), (11,12), (5,13), (13,14), (14,15)]
left_keypoint = [3,4,9,13,14,15]
right_keypoint = [1,2,8,10,11,12]
mid_keypoint = [0,5,6,7]

left_color = np.array((0, 240, 0))/255
right_color = np.array((250, 126, 0))/255
mid_color = np.array((52, 143, 235))/255

COLOR_SET = [(145, 226, 88), (245, 88, 64), (237, 147, 181), (39, 245, 236), (253, 245, 81), (191, 189, 237), \
    (249, 115, 230), (109, 255, 154), (242, 221, 210), (159, 217, 214), (240, 238, 236), (244, 177, 131), \
    (214, 255, 121), (237, 141, 139), (123, 182, 253), (255, 223, 121), (121, 255, 214), (255, 121, 121), \
    (188, 133, 243), (203, 220, 156), (126, 209, 250), (249, 255, 121), (172, 255, 121), (222, 235, 247)]
INVIS_COLOR = (np.array((251, 229, 214))/255.0).tolist()

BOX_VIS_SET = {
    'Visble': 0,
    'Truncated': 1,
    'Occluded': 2,
}


def parse_opt():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument(
        '-i', '--input', help='path of COCO-VID formatted ChimpACT data', default='data/ChimpACT_processed')
    parser.add_argument(
        '-o', '--output', help='path of saved data', default='tools/vis_annots')
    parser.add_argument(
        '--vid-name', help='video name to be visualized', default='Azibo_ObsChimp_2017_06_22_c_clip_44000_45000')
    parser.add_argument(
        '--interval', help='sampling interval of results to be visualizaed', default=10)
    # visualize args
    parser.add_argument('--vis-action', action='store_true', help='visualize action labels')
    parser.add_argument('--vis-tracking', action='store_true', help='visualize tracking labels')
    parser.add_argument('--vis-pose', action='store_true', help='visualize pose labels')
    parser.add_argument('--save-img', action='store_true', help='save images')

    opt = parser.parse_args()
    return opt


def show_annots(img, anns, frame_idx, draw_tracking=True, draw_action=False, draw_pose=None):
    if len(anns) == 0:
        return img, img, -2
    
    iskeyframe = anns[0]['iskeyframe']
    if not iskeyframe and draw_pose:
        return img, img, -2
    ori_img = img.copy()

    img_h, img_w = img.shape[:2]
    txt_w_min, txt_h_min = 10, 10

    color_text = (0, 0, 0)
    color_bg = (233, 235, 235)
    for ann in anns:
        bbox_id = ann['instance_id']
        color_box = tuple(map(int, np.array(COLOR_SET[bbox_id])))
        bbox_name = NAME_SET[bbox_id][0]
        bbox_vis = ann['visibility']
        behave_list = ann['behaviors']
        if bbox_name == '':
            bbox_name = '?'


        if draw_pose and 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            kps = np.array(ann['keypoints'])
            if kps.shape[0] != 48:
                return ori_img, img, frame_idx
            kps_x, kps_y, kps_vis = (kps[0::3]).astype(np.int32), (kps[1::3]).astype(np.int32), kps[2::3]
            vis = kps_vis
            poses = np.concatenate([kps_x[:,None], kps_y[:,None]],-1)
            for i, jt in enumerate(skeleton):
                if jt in LEFT_SKELE:
                    color = left_color
                elif jt in RIGHT_SKELE:
                    color = right_color
                else:
                    color = mid_color
                xs, ys = poses[jt[0]], poses[jt[1]]
                xvis, yvis = vis[jt[0]], vis[jt[1]]
                if xvis == 0 or yvis == 0:
                    continue
                cv2.line(img, xs, ys, color*255.0, 2)


        if draw_tracking or draw_action:
            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            bbox_x, bbox_y, bbox_w, bbox_h = int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)

            cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x+bbox_w, bbox_y+bbox_h), color_box, 1)

            txt_attrs = ''
            if draw_tracking:
                txt_attrs += f"{bbox_id} | {bbox_name}"
                bg_offset = 130
            if draw_action:
                bg_offset = 160
                behaviors = ''
                for behave in behave_list:
                    behaviors += f'{BEHAVE_ID_SET[behave]}, '
                if draw_tracking:
                    bg_offset += 100
                    txt_attrs += ' | '
                txt_attrs += f"{behaviors[:-2]}"
            txt_size = cv2.getTextSize(f"{txt_attrs}", 0, 0.5, 1)[0]
            txt_w, txt_h = txt_size[:2]
            txt_w_max, txt_h_max = img_w-txt_w, img_h-txt_h
            txt_x, txt_y = min(max(bbox_x-10, txt_w_min), txt_w_max), min(max(bbox_y-10, txt_h_min), txt_h_max)
            cv2.rectangle(img, (txt_x+5, txt_y-30), (txt_x+txt_w+bg_offset, txt_y+txt_h), color_bg, thickness=-1)
            cv2.putText(img, f"{txt_attrs}", (txt_x+10, txt_y), cv2.FONT_HERSHEY_DUPLEX, 1, color_text, 2)

    return ori_img, img, -1


def run(opt):

    save_dir = Path(opt.output)  # save path for visualization
    os.makedirs(save_dir, exist_ok=True)  # make dir
    if opt.save_img:
        os.makedirs(osp.join(save_dir, 'imgs'), exist_ok=True)  # make dir

    vis_action = opt.vis_action
    vis_tracking = opt.vis_tracking
    vis_pose = opt.vis_pose
    vis_vid_name = opt.vid_name
    interval = int(opt.interval)
    for subset in ['train', 'val', 'test']:
        annot_file = CocoVID(osp.join(opt.input, 'annotations', f'{subset}.json'))
        vid_ids = annot_file.get_vid_ids()
        for vid_id in tqdm(vid_ids):
            cat_ids = annot_file.get_cat_ids()
            img_ids = annot_file.get_img_ids_from_vid(vid_id)
            vid_name = annot_file.load_vids([vid_id])[0]['name'][:-4]
        
            vis_path = osp.join(opt.output, f'{vid_name}_annotated.mp4')
            videowriter = imageio.get_writer(vis_path, fps=25)
            for img_id in tqdm(img_ids):
                img_info = annot_file.load_imgs([img_id])[0]
                img_path = osp.join(opt.input, subset, 'images', img_info['file_name'])
                vid_name = img_info['file_name'].split('/')[-2]
                img_name_idx = int(img_info['file_name'].split('/')[-1][:-4])
                if vid_name != vis_vid_name:
                    continue

                img = np.array(cv2.imread(img_path)[...,::-1])
                frame_idx = img_info['frame_id']

                ann_ids = annot_file.get_ann_ids(img_ids=[img_id], cat_ids=cat_ids)
                objs = annot_file.load_anns(ann_ids)
                ori_img, img, flg = show_annots(img, objs, frame_idx, draw_tracking=vis_tracking, draw_action=vis_action, draw_pose=vis_pose)
                if opt.save_img and not img_name_idx % interval:
                    save_img_path = osp.join(save_dir, 'imgs', f'gt_{vid_name}_{img_name_idx}.png')
                    cv2.imwrite(save_img_path, img[...,::-1])
                if flg == -2:
                    continue
                videowriter.append_data(np.concatenate([ori_img, img], 1))

            videowriter.close()


def main(opt):
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)