import json
import numpy as np
import argparse
import os
import os.path as osp
import pandas as pd
from pandas import DataFrame
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
np.set_printoptions(suppress=True)
from mmtrack.datasets.parsers import CocoVID

BEHAVE_ID_DICT = {
    0: 'moving',
    1: 'climbing',
    2: 'resting',
    3: 'sleeping',
    4: 'solitary object playing',
    5: 'eating',
    6: 'manipulating object',
    7: 'grooming',
    8: 'being groomed',
    9: 'aggressing',
    10: 'embracing',
    11: 'begging',
    12: 'being begged from',
    13: 'taking object',
    14: 'losing object',
    15: 'carrying',
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


def parse_opt():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument(
        '-i', '--input', help='path of COCO-VID formatted ChimpACT data', default='data/ChimpACT_processed')
    parser.add_argument(
        '-o', '--output', help='path of saved data', default='tools/vis_stats')
    opt = parser.parse_args()
    return opt

def vis_social_violin(nameid_social_arr, save_dir):
    scale = 'width'
    scale_hue = True

    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    nameid_social_arr = pd.DataFrame(np.array(nameid_social_arr), columns=['Chimpanzee names', 'Behaviors'])
    nameid_social_arr.drop(nameid_social_arr[nameid_social_arr['Chimpanzee names'] == 23].index, inplace = True)
    nameid_social_arr = nameid_social_arr.reset_index()
    len(nameid_social_arr[(nameid_social_arr['Behaviors']==3)].index.tolist())
    sns.set(font_scale=1.2)
    sns.violinplot(x="Chimpanzee names", y="Behaviors", data=nameid_social_arr, linewidth=0, scale=scale, scale_hue=scale_hue, inner='point')
    namelist = list(NAME_SET_REV.keys())
            
    plt.xticks(list(range(23)), namelist[:23], rotation=45)
    plt.ylim((-1,14))
    plt.yticks(list(range(14)), list(BEHAVE_SOCIAL.keys()), rotation=45)
    plt.ylabel('Behaviors', fontsize=16)
    plt.subplots_adjust(left=0.23, right=0.96, top=0.97, bottom=0.18)
    plt.xlabel('Chimpanzee names', fontsize=16)
    plt.savefig(osp.join(save_dir, 'supp_dist_behavior_social_identity.pdf'), dpi=600)

def vis_identity_dist(nameid_arr, save_dir):
    plt.close('all')
    fig = plt.figure(figsize=(18, 6))
    nameid_pd = {'Chimpanzees': [], 'Frequency': []}
    all_counts = 0
    for name, count in nameid_arr.items():
        if name == '':
            all_counts += count
            continue
        nameid_pd['Chimpanzees'].append(name)
        nameid_pd['Frequency'].append(count)
        all_counts += count
    for cid, count in enumerate(nameid_pd['Frequency']):
        nameid_pd['Frequency'][cid] = round(nameid_pd['Frequency'][cid]/all_counts*100, 3) 
    ax = sns.barplot(data=DataFrame(nameid_pd), x="Chimpanzees", y="Frequency", dodge=False, order=DataFrame(nameid_pd).sort_values('Frequency', ascending=False).Chimpanzees)
    ax.bar_label(ax.containers[0])
    plt.xticks(list(range(23)), DataFrame(nameid_pd).sort_values('Frequency', ascending=False).Chimpanzees, rotation=45)
    plt.yscale('log')
    # ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%'])
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25)
    plt.xlabel('Chimpanzee names', fontsize=16)
    plt.ylabel('Frequency (%)', fontsize=16)
    plt.savefig(osp.join(save_dir, 'dist_identity_log.pdf'), dpi=600)

def vis_behavior_dist(behavior_arr, save_dir):
    plt.close('all')
    fig = plt.figure(figsize=(18, 6))
    behave_pd = {'Behaviors': [], 'Frequency': []}
    all_counts = 0
    for name, count in behavior_arr.items():
        behave_pd['Behaviors'].append(BEHAVE_ID_DICT[name])
        behave_pd['Frequency'].append(count)
        all_counts += count
    for cid, count in enumerate(behave_pd['Frequency']):
        behave_pd['Frequency'][cid] = round(behave_pd['Frequency'][cid]/all_counts*100, 3) 
    ax = sns.barplot(data=DataFrame(behave_pd), x="Behaviors", y="Frequency", dodge=False, order=DataFrame(behave_pd).sort_values('Frequency', ascending=False).Behaviors)
    ax.bar_label(ax.containers[0])
    plt.xticks(list(range(23)), DataFrame(behave_pd).sort_values('Frequency', ascending=False).Behaviors, rotation=45)
    plt.yscale('log')
    # ax.set_yticklabels(['0.01%', '0.1%', '0.01%', '0.1%', '1%', '10%'])
    ax.set_ylim(0,50)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25)
    plt.xlabel('Behaviors', fontsize=16)
    plt.ylabel('Frequency (%)', fontsize=16)
    plt.savefig(osp.join(save_dir, 'dist_behavior_log.pdf'), dpi=600)

def run(opt):

    frames_num_tracking = 0
    frames_num_pose = 0
    label_num_tracking = 0
    label_num_pose = 0
    label_num_action = 0
    label_num_locomotion = 0
    label_num_object = 0
    label_num_social = 0
    label_num_others = 0
    label_num_each_action = np.zeros(23)
    behavior_arr = defaultdict(int)
    nameid_arr = defaultdict(int)
    nameid_social_arr = []

    for subset in ['train', 'val', 'test']:
        annot_file = CocoVID(osp.join(opt.input, 'annotations', f'{subset}.json'))
        vid_ids = annot_file.get_vid_ids()

        for vid_id in tqdm(vid_ids):
            cat_ids = annot_file.get_cat_ids()
            img_ids = annot_file.get_img_ids_from_vid(vid_id)

            for img_id in tqdm(img_ids):
                ann_ids = annot_file.get_ann_ids(img_ids=[img_id], cat_ids=cat_ids)
                anns = annot_file.load_anns(ann_ids)
                frames_num_tracking += 1

                if len(anns) == 0:
                    continue
                iskeyframe = anns[0]['iskeyframe']
                if iskeyframe:
                    frames_num_pose += 1
                for ann in anns:
                    if iskeyframe:
                        label_num_tracking += 1
                        nameid_arr[NAME_SET[ann['instance_id']][0]] += 1
                    if 'keypoints' in ann and type(ann['keypoints']) == list:
                        if np.array(ann['keypoints']).shape[0] == 48:
                            label_num_pose += 1
                    if iskeyframe:
                        for behave in ann['behaviors']:
                            behavior_arr[behave] += 1
                            label_num_action += 1
                            if behave in locomotion_ids:
                                label_num_locomotion += 1
                            if behave in object_ids:
                                label_num_object += 1
                            if behave in social_ids:
                                label_num_social += 1
                                nameid_social_arr.append([ann['instance_id'], behave-7])
                            if behave in others_ids:
                                label_num_others += 1
                            label_num_each_action[int(behave)]+=1
                            

    print('# Frames of <Track 1: Tracking> ', frames_num_tracking)    # 160500
    print('# Boxes of <Track 1: Tracking> ', label_num_tracking)      # 56324
    print('# Frames of <Track 2: Pose> ', frames_num_pose)            # 16028
    print('# Poses of <Track 2: Pose> ', label_num_pose)              # 56324
    print('# Labels of <Track 3: Action> ', label_num_action)         # 64289
    print('------')
    print(f'# ##### Percentage of [Locomotion] labels in <Track 3: Action> {label_num_locomotion/label_num_action*100:.2f}%')       # 45.01%
    print(f'# ##### Percentage of [Object Interaction] labels in <Track 3: Action> {label_num_object/label_num_action*100:.2f}%')   # 20.17%
    print(f'# ##### Percentage of [Social Interaction] labels in <Track 3: Action> {label_num_social/label_num_action*100:.2f}%')   # 34.68%
    print(f'# ##### Percentage of [Others] labels in <Track 3: Action> {label_num_others/label_num_action*100:.2f}%')               # 0.15%
    print('------')
    for bidx in range(23):
        print(f'# ##### Percentage of [{BEHAVE_ID_DICT[bidx]}] labels in <Track 3: Action> {label_num_each_action[bidx]/label_num_action*100:.2f}%')
    print('------')

    os.makedirs(opt.output, exist_ok=True)
    vis_identity_dist(nameid_arr, save_dir=opt.output)
    vis_behavior_dist(behavior_arr, save_dir=opt.output)
    vis_social_violin(nameid_social_arr, save_dir=opt.output)


def main(opt):
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)