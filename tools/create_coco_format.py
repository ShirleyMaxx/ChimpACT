# This script integrates ChimpACT original labels into a single COCO-VID-style annotation file.
#
# NOTE! We label one frame every 10 frames, this script propagates the original labels to the full frames by interpolation.
# NOTE! We use `iskeyframe` to distinguish the manual annotations from the propagated annotations.
#
# ##############################################

import argparse
import os
import os.path as osp
from collections import defaultdict
from termcolor import colored
import mmcv
import numpy as np
import cv2
from tqdm import tqdm
import math
import subprocess
from pycocotools.coco import COCO

# validation set
val_vid_name_list = [
    'Azibo_ObsChimp_2015_11_25_d_clip_23000_24000',
    'Azibo_ObsChimp_2015_11_26_a_clip_1000_2000',
    'Azibo_ObsChimp_2016_08_02_c_clip_32000_33000',
    'Azibo_ObsChimp_2017_02_27_a_clip_13000_14000',
    'Azibo_ObsChimp_2017_11_10_clip_7000_8000',
    'Azibo_ObsChimp_2017_11_10_clip_8000_9000',
    'Azibo_ObsChimp_2017_06_22_c_clip_46000_47000',
    'Azibo_ObsChimp_2017_06_22_c_clip_67000_68000',
    'Azibo_ObsChimp_2018_07_11_c_clip_0_1000',
    'Azibo_ObsChimp_2018_07_11_c_clip_1000_2000',
    'Azibo_ObsChimp_2018_07_11_c_clip_3000_4000',
    'Azibo_ObsChimp_2018_07_11_c_clip_6000_7000',
    'Azibo_ObsChimp_2018_07_11_c_clip_17000_18000',
    'Azibo_ObsChimp_2018_07_11_c_clip_18000_19000',
    'Azibo_ObsChimp_2018_08_06_a_clip_7000_8000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_15000_16000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_16000_17000',
]

test_vid_name_list = [
    'Azibo_ObsChimp_2015_11_25_d_clip_1000_2000',
    'Azibo_ObsChimp_2015_11_26_a_clip_0_1000',
    'Azibo_ObsChimp_2015_11_26_a_clip_2000_3000',
    'Azibo_ObsChimp_2016_08_02_c_clip_33000_34000',
    'Azibo_ObsChimp_2016_08_15_b_clip_2000_3000',
    'Azibo_ObsChimp_2016_10_27_c_clip_0_1000',
    'Azibo_ObsChimp_2017_02_27_a_clip_14000_15000',
    'Azibo_ObsChimp_2017_11_10_clip_6000_7000',
    'Azibo_ObsChimp_2017_06_22_c_clip_44000_45000',
    'Azibo_ObsChimp_2017_06_22_c_clip_68000_69000',
    'Azibo_ObsChimp_2018_07_06_d_clip_0_696',
    'Azibo_ObsChimp_2018_07_11_c_clip_2000_3000',
    'Azibo_ObsChimp_2018_07_11_c_clip_8000_9000',
    'Azibo_ObsChimp_2018_07_11_c_clip_16000_17000',
    'Azibo_ObsChimp_2018_07_11_c_clip_19000_20000',
    'Azibo_ObsChimp_2018_08_06_a_clip_6000_7000',
    'Azibo_ObsChimp_2018_08_06_a_clip_8000_9000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_14000_15000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_17000_17712',
]

test_indoor_vid_name_list = [
    'Azibo_ObsChimp_2015_11_25_d_clip_1000_2000',
    'Azibo_ObsChimp_2015_11_26_a_clip_0_1000',
    'Azibo_ObsChimp_2015_11_26_a_clip_2000_3000',
    'Azibo_ObsChimp_2016_08_15_b_clip_2000_3000',
    'Azibo_ObsChimp_2016_10_27_c_clip_0_1000',
    'Azibo_ObsChimp_2017_02_27_a_clip_14000_15000',
    'Azibo_ObsChimp_2017_11_10_clip_6000_7000',
    'Azibo_ObsChimp_2017_06_22_c_clip_44000_45000',
    'Azibo_ObsChimp_2017_06_22_c_clip_68000_69000',
    'Azibo_ObsChimp_2018_07_06_d_clip_0_696',
    'Azibo_ObsNatascha_2018_06_29_a_clip_14000_15000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_17000_17712',
]

test_outdoor_vid_name_list = [
    'Azibo_ObsChimp_2016_08_02_c_clip_33000_34000',
    'Azibo_ObsChimp_2018_07_11_c_clip_2000_3000',
    'Azibo_ObsChimp_2018_07_11_c_clip_8000_9000',
    'Azibo_ObsChimp_2018_07_11_c_clip_16000_17000',
    'Azibo_ObsChimp_2018_07_11_c_clip_19000_20000',
    'Azibo_ObsChimp_2018_08_06_a_clip_6000_7000',
    'Azibo_ObsChimp_2018_08_06_a_clip_8000_9000',
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ChimpACT label to COCO-VID format.')
    parser.add_argument(
        '-i', '--input', help='path of ChimpACT data', default='data/ChimpACT_release')
    parser.add_argument(
        '-o', '--output', help='path to save COCO-VID formatted label file', default='data/ChimpACT_processed')
    return parser.parse_args()


def video2frames(vid_path, save_dir=None):
    cap = cv2.VideoCapture(vid_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    vid_name = vid_path.split('/')[-1][:-4]
    subset = 'train'
    if vid_name in val_vid_name_list:
        subset = 'val'
    elif vid_name in test_vid_name_list:
        subset = 'test'
    img_dir = osp.join(save_dir, subset, 'images', vid_name)
    os.makedirs(img_dir, exist_ok=True)
    vid_dir = osp.join(save_dir, subset, 'videos', f'{vid_name}.mp4')
    try:
        os.symlink(osp.join(os.getcwd(), vid_path), osp.join(os.getcwd(), vid_dir))
    except:
        print('Warning! Symbolic link for video already exists.')
    command = ['ffmpeg',
               '-i', vid_path,
               '-r', str(fps),
               '-f', 'image2',
               '-v', 'error',
               '-start_number', '0',
               f'{img_dir}/%06d.jpg']
        
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    print(f'Images saved to \"{img_dir}\"')
    return subset, vid_dir


def main():
    args = parse_args()
    data_dir = args.input
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    video_dir = osp.join(data_dir, 'videos_full')
    label_dir = osp.join(data_dir, 'labels')

    # ######## video2images ########
    print('===> clipping videos to frames ...')
    for subset in ['train', 'val', 'test']:
        os.makedirs(osp.join(save_dir, subset, 'images'), exist_ok=True)
        os.makedirs(osp.join(save_dir, subset, 'videos'), exist_ok=True)
    video_path_dict = {'train': [], 'val': [], 'test': [], 'test_indoor': [], 'test_outdoor': []}
    video_name_list = sorted(os.listdir(video_dir))
    for video_name in tqdm(video_name_list):
        video_path = osp.join(video_dir, video_name)
        subset, video_path_new = video2frames(video_path, save_dir)
        video_path_dict[subset].append(video_path_new)
    for keys in ['test_indoor', 'test_outdoor']:
        for vid_name in eval(f'{keys}_vid_name_list'):
            video_path_dict[keys].append(osp.join(save_dir, 'test', 'videos', f'{vid_name}.mp4'))
    # ######## labels2coco ########
    anno_dir = osp.join(save_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    print('===> converting labels to COCO VID format ...')
    vid_id, img_id, ann_id = 1, 1, 1
    for subset in ['train', 'val', 'test', 'test_indoor', 'test_outdoor']:
        print(colored(f'\n  ==============  processing {subset} set ==============', 'green'))
        
        out_file = osp.join(anno_dir, f'{subset}.json')
        outputs = defaultdict(list)
        outputs['categories'] = [dict(id=1, name='chimpanzee')]

        for video_path in tqdm(video_path_dict[subset]):
            video_name = video_path.split('/')[-1]
            print(video_name, video_path)
            # whether a clip is 1000 vs. 100
            not_full_flag = False

            # video-level infos
            vid_cap = cv2.VideoCapture(video_path)
            frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   
            fps = vid_cap.get(cv2.CAP_PROP_FPS)  
            video = dict(
                id=vid_id,
                name=video_name,
                fps=fps,
                frames=frames,
                width=width,
                height=height)
            outputs['videos'].append(video)

            # image-level sanity check (original)
            img_folder = video_path.replace('videos', 'images').replace('.mp4', '')
            img_names = [f for f in os.listdir(f'{img_folder}') if f.endswith('.jpg')]
            img_names = sorted(img_names)
            assert frames == len(img_names)

            # parse annotations (downsample x10)
            label_path = osp.join(label_dir, f'{video_name[:-4]}.json')
            labels = COCO(label_path) # # dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])
            image_set_index = labels.getImgIds()
            num_images = len(image_set_index)
            frame_index = list(np.array(image_set_index) - image_set_index[0])
            actual_key_frame_id = [int(labels.loadImgs(key_ann_id)[0]['file_name'].split('/')[-1].split('.')[0]) for key_ann_id in image_set_index]

            full_image_set_index = np.zeros(math.ceil(frames/10), dtype=np.int32)-1
            full_image_set_index[actual_key_frame_id] = image_set_index
            full_image_set_index = full_image_set_index.tolist()

            # image and box level infos
            actual_frame_id = 0
            for frame_id, name in enumerate(img_names):
                key_frame_id = frame_id//10
                if full_image_set_index[key_frame_id] == -1:
                    continue
                img_name = osp.join(video_name[:-4], name)
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=actual_frame_id)
                actual_frame_id += 1
                outputs['images'].append(image)

                # instance-level info
                if frame_id%10 == 0:        # keyframe
                    key_ann_id = full_image_set_index[key_frame_id]
                
                    im_ann = labels.loadImgs(key_ann_id)[0]

                    annIds = labels.getAnnIds(imgIds=key_ann_id, iscrowd=False)
                    objs = labels.loadAnns(annIds)

                    for obj in objs:
                        bbox = obj['bbox']
                        anns = dict(
                            id=ann_id,
                            video_id=vid_id,
                            image_id=img_id,
                            category_id=1,
                            bbox=bbox,
                            area=bbox[2] * bbox[3],
                            iscrowd=False,
                            visibility=obj['bbox_vis'],
                            behaviors=obj['behaviors'],
                            keypoints=obj['keypoints'],
                            iskeyframe=1,
                            instance_id=obj['bbox_id'])
                        outputs['annotations'].append(anns)
                        ann_id += 1
                else:       # not keyframe, labels propagated from keyframes
                    ref_key_frame_id = key_frame_id
                    ref_key_frame_id_next = ref_key_frame_id + 1
                    ref_key_ann_id = full_image_set_index[ref_key_frame_id]
                    try:
                        ref_key_ann_id_next = full_image_set_index[ref_key_frame_id_next]
                    except:
                        ref_key_ann_id_next = ref_key_ann_id
                    
                    ref_annIds = labels.getAnnIds(imgIds=ref_key_ann_id, iscrowd=False)
                    ref_objs = labels.loadAnns(ref_annIds)

                    ref_annIds_next = labels.getAnnIds(imgIds=ref_key_ann_id_next, iscrowd=False)
                    ref_objs_next = labels.loadAnns(ref_annIds_next)

                    for ref_obj in ref_objs:
                        bbox = ref_obj['bbox']
                        instance_id = ref_obj['bbox_id']

                        find_next = -1
                        for ref_id_next, ref_obj_next in enumerate(ref_objs_next):
                            if ref_obj_next['bbox_id'] == instance_id:
                                find_next = ref_id_next
                                break

                        if find_next == -1:
                            # disappear in next key frame
                            anns = dict(
                                id=ann_id,
                                video_id=vid_id,
                                image_id=img_id,
                                category_id=1,
                                bbox=bbox,
                                area=bbox[2] * bbox[3],
                                iscrowd=False,
                                visibility=2,   # 'occluded' for not sure
                                behaviors=ref_obj['behaviors'],
                                keypoints=[],
                                iskeyframe=0,   # not key frame
                                instance_id=instance_id)
                        else:
                            # interpolate between two key frames
                            assert ref_obj_next == ref_objs_next[find_next]
                            bbox_next = ref_obj_next['bbox']

                            new_bbox = np.array(bbox)*(1-frame_id%10/10) + np.array(ref_obj_next['bbox'])*(frame_id%10/10)
                            new_behaviors = ref_obj['behaviors'] if frame_id%10/10 <= 0.5 else ref_obj_next['behaviors']
                            new_bbox_vis = ref_obj['bbox_vis'] if frame_id%10/10 <= 0.5 else ref_obj_next['bbox_vis']
                            anns = dict(
                                id=ann_id,
                                video_id=vid_id,
                                image_id=img_id,
                                category_id=1,
                                bbox=new_bbox,
                                area=new_bbox[2] * new_bbox[3],
                                iscrowd=False,
                                visibility=new_bbox_vis,   # 
                                behaviors=new_behaviors,
                                keypoints=[],
                                iskeyframe=0,   # not key frame
                                instance_id=instance_id)
                        outputs['annotations'].append(anns)
                        ann_id += 1

                img_id += 1

            vid_id += 1
 

        print(f"  ======>  {subset} set has {len(outputs['annotations'])} instances, {img_id} images.")
        mmcv.dump(outputs, out_file)
        print(f'  ======>  Done! Saved as {out_file}')


if __name__ == '__main__':
    main()
