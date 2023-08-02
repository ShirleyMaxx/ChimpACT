All the instructions should be in the `chimp_track` virtual environment and under the `mmtracking` directory.
```bash
conda activate chimp_track
cd mmtracking
```

# Train
```python
# Train detection & reid models for SORT, DeepSORT, Tracktor
# Train detection model with 4 GPUs
bash tools/dist_train.sh configs/det/faster-rcnn_r50_fpn_chimp.py 4
bash tools/dist_train.sh configs/det/yolox_x_chimp.py 4

# Train reid model with 4 GPUs
bash tools/dist_train.sh configs/reid/reid_resnet50_b32x8_chimp.py 4

# Train tracking model with 4 GPUs
bash tools/dist_train.sh configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_chimp.py 4
bash tools/dist_train.sh configs/mot/bytetrack/bytetrack_faster-rcnn_r50_fpn_chimp.py 4
bash tools/dist_train.sh configs/mot/bytetrack/bytetrack_yolox_x_chimp.py 4
bash tools/dist_train.sh configs/mot/ocsort/ocsort_faster-rcnn_r50_fpn_chimp.py 4
bash tools/dist_train.sh configs/mot/ocsort/ocsort_yolox_x_chimp.py 4
```

# Eval AP
```python
# Test tracking model with 4 GPUs
# SORT
bash tools/dist_test.sh configs/mot/deepsort/sort_faster-rcnn_fpn_4e_chimp.py 4  --eval track bbox
bash tools/dist_test.sh configs/mot/deepsort/sort_yolox_x_chimp.py 4  --eval track bbox

# DeepSORT
bash tools/dist_test.sh configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_chimp.py 4  --eval track bbox
bash tools/dist_test.sh configs/mot/deepsort/deepsort_yolox_x_chimp.py 4  --eval track bbox

# Tracktor
bash tools/dist_test.sh configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_chimp.py 4  --eval track bbox

# QDTrack
bash tools/dist_test.sh configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_chimp.py 4 --checkpoint work_dirs/qdtrack_faster-rcnn_r50_fpn_4e_chimp/latest.pth --eval track bbox

# ByteTrack
bash tools/dist_test.sh configs/mot/bytetrack/bytetrack_faster-rcnn_r50_fpn_chimp.py 4 --checkpoint work_dirs/bytetrack_faster-rcnn_r50_fpn_chimp/latest.pth --eval track bbox
bash tools/dist_test.sh configs/mot/bytetrack/bytetrack_yolox_x_chimp.py 4 --checkpoint work_dirs/bytetrack_yolox_x_chimp/latest.pth --eval track bbox

# OC-SORT
bash tools/dist_test.sh configs/mot/ocsort/ocsort_faster-rcnn_r50_fpn_chimp.py 4 --checkpoint work_dirs/ocsort_faster-rcnn_r50_fpn_chimp/latest.pth  --eval track bbox
bash tools/dist_test.sh configs/mot/ocsort/ocsort_yolox_x_chimp.py 4 --checkpoint work_dirs/ocsort_yolox_x_chimp/latest.pth --eval track bbox
```

# Visualization
- Use below shell scripts to visualize the tracking results. An example is given.

```bash
# this script will visualize the tracking results of the video clip 'Azibo_ObsChimp_2017_06_22_c_clip_44000_45000.mp4' by 'deepsort_faster-rcnn' model.

vid_list=('Azibo_ObsChimp_2017_06_22_c_clip_44000_45000.mp4')
for vid_name in "${vid_list[@]}"
do
echo $vid_name
python demo/demo_mot_vis.py \
    configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_chimp.py \
    --input data/ChimpACT_processed/test/videos/$vid_name \
    --output work_dirs/vis_results/deepsort_faster-rcnn_fpn_4e_chimp/$vid_name \
    --checkpoint work_dirs/faster-rcnn_r50_fpn_chimp/latest.pth
done
```
