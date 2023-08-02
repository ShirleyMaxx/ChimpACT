# This script create MOT datasets based on the processed COCO-VID-style annotation file.
#
# ##############################################

import argparse
import os
import os.path as osp
import random
import multiprocessing as mp

import json
import mmcv
import numpy as np
from tqdm import tqdm

id_dicts = {i: i - 1 if i < 8 else i - 2 if i > 8 else -1 for i in range(1, 24)} # remove instance_id==8, id should start from 0

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create MOT dataset and ReID dataset.')
    parser.add_argument(
        '-i', '--input', default='data/ChimpACT_processed', help='path of the dataset')
    parser.add_argument(
        '-o', '--output', default='data/ChimpACT_processed/reid', help='path to save ReID dataset')
    parser.add_argument(
        '--vis-threshold', type=float, default=0.3, help='threshold of visibility for each person')
    parser.add_argument(
        '-p', '--processes', type=int, default=1, help='number of processes in process-based parallelism')

    return parser.parse_args()



def process_mot_dataset(args):
    data_dir = args.input
    for subset in ['train', 'val', 'test']:
        data = json.load(open(osp.join(data_dir, f'annotations/{subset}.json')))
        annotations = data['annotations']
        img_annot_dict = {annot['id']: annot for annot in data['images']}
        video_annotations = {}
        for annot in annotations:
            image_id, instance_id, bbox = annot['image_id'], annot['instance_id'], annot['bbox']
            img_annot = img_annot_dict[image_id]
            video_name, frame_id = img_annot['file_name'].split('/')[0], img_annot['frame_id']
            
            if video_name not in video_annotations:
                video_annotations[video_name] = []
            video_annotations[video_name].append({
                'id': instance_id,
                'frame_id': frame_id,
                'bbox': bbox,
            })

        for video_name, annots in video_annotations.items():
            annots = sorted(annots, key=lambda x: (x['id'], x['frame_id']))

            gt_path = osp.join(data_dir, f'{subset}/images/{video_name}/gt')
            os.makedirs(gt_path, exist_ok=True)
            annots = sorted(annots, key=lambda x: (x['id'], x['frame_id']))
            with open(osp.join(gt_path, 'gt.txt'), 'w') as f:
                for annot in annots:
                    instance_id = annot['id'] + 1 if annot['id'] != 23 else -1
                    if instance_id == -1:
                        continue
                    f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,0,1\n'%(annot['frame_id'] + 1, instance_id, annot['bbox'][0], annot['bbox'][1], annot['bbox'][2], annot['bbox'][3]))

            seq_info_path = osp.join(data_dir, f'{subset}/images/{video_name}/seqinfo.ini')
            for data_video_anno in data['videos']:
                if data_video_anno['name'][:-4] == video_name:
                    break
            with open(seq_info_path, 'w') as f:
                info = f'[Sequence]\nname={video_name}\nimDir=\nframeRate={int(data_video_anno["fps"])}\nseqLength={data_video_anno["frames"]}\nimWidth={data_video_anno["width"]}\nimHeight={data_video_anno["height"]}\nimExt=.jpg'
                f.writelines(info)


def process_images(video_name, in_folder, args):
    # load video infos
    video_folder = osp.join(in_folder, video_name)
    infos = mmcv.list_from_file(f'{video_folder}/seqinfo.ini')
    # video-level infos
    assert video_name == infos[1].strip().split('=')[1]
    raw_img_folder = infos[2].strip().split('=')[1]
    imext = infos[7].strip().split('=')[1]
    raw_img_names = os.listdir(f'{video_folder}/{raw_img_folder}')
    raw_img_names = sorted([img_name for img_name in raw_img_names if img_name.endswith(imext)])
    num_raw_imgs = int(infos[4].strip().split('=')[1])
    assert num_raw_imgs == len(raw_img_names)

    reid_train_folder = osp.join(args.output, 'imgs')
    if not osp.exists(reid_train_folder):
        os.makedirs(reid_train_folder)
    gts = mmcv.list_from_file(f'{video_folder}/gt/gt.txt')
    last_frame_id = -1
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        ins_id = id_dicts[ins_id]
        if ins_id < 0: continue
        ltwh = list(map(float, gt[2:6]))
        class_id = int(gt[7])
        visibility = float(gt[8])
      
        if visibility < args.vis_threshold:
            continue
        reid_img_folder = osp.join(reid_train_folder,
                                    f'{video_name}_{ins_id:06d}')
        if not osp.exists(reid_img_folder):
            os.makedirs(reid_img_folder)
        idx = len(os.listdir(reid_img_folder))
        reid_img_name = f'{idx:06d}.jpg'
        if frame_id != last_frame_id:
            raw_img_name = raw_img_names[frame_id - 1]
            raw_img = mmcv.imread(
                f'{video_folder}/{raw_img_folder}/{raw_img_name}')
            last_frame_id = frame_id
        xyxy = np.asarray(
            [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]])
        reid_img = mmcv.imcrop(raw_img, xyxy)
        mmcv.imwrite(reid_img, f'{reid_img_folder}/{reid_img_name}')


def process_reid_dataset(args):
    if not osp.isdir(args.output):
        os.makedirs(args.output)
    elif os.listdir(args.output):
        raise OSError(f'Directory must empty: \'{args.output}\'')
    
    reid_folder = osp.join(args.output, f'imgs')
    if not osp.exists(reid_folder):
        os.makedirs(reid_folder)
    
    pool = mp.Pool(processes=args.processes)

    for subset in ['train', 'val', 'test']:
        in_folder = osp.join(args.input, f'{subset}/images')

        video_names = os.listdir(in_folder)
        
        jobs = [pool.apply_async(process_images, (video_name, in_folder, args)) for video_name in video_names]
        [job.get() for job in tqdm(jobs)]
    
        
        reid_meta_folder = osp.join(args.output, 'meta')
        if not osp.exists(reid_meta_folder):
            os.makedirs(reid_meta_folder)
        reid_list = []
        reid_img_folder_names = sorted(os.listdir(reid_folder))
        random.seed(0)

        for reid_img_folder_name in reid_img_folder_names: 
            reid_img_names = os.listdir(
                f'{reid_folder}/{reid_img_folder_name}')
            
            label = int(reid_img_folder_name.split('_')[-1])
            if label < 0: continue
            for reid_img_name in reid_img_names:
                reid_list.append(
                    f'{reid_img_folder_name}/{reid_img_name} {label}\n')

        reid_entire_dataset_list = reid_list.copy()

        with open(osp.join(reid_meta_folder, f'{subset}.txt'), 'w') as f:
            f.writelines(reid_entire_dataset_list)


if __name__ == '__main__':
    args = parse_args()
    process_mot_dataset(args)
    process_reid_dataset(args)
