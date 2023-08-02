All the instructions should be in the `chimp_pose` virtual environment and under the `mmpose` directory.
```bash
conda activate chimp_pose
cd mmpose
```

# Train
```python
# Train regression-based methods with 4 GPUs
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res101_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res152_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_mobilenetv2_rle-pretrained-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res101_rle-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res152_rle-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000

# Train heatmap-based methods with 4 GPUs
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_cpm_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hourglass52_8xb32-210e_coco-256x256.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-small_8xb32-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
bash tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-256x192.py 4 --show-dir vis_pose --interval 1000
```

# Eval AP
```python
# Eval regression-based methods with 4 GPUs and save predictions to file
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py work_dirs/td-reg_res50_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_res50.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res101_8xb64-210e_coco-256x192.py work_dirs/td-reg_res101_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_res101.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res152_8xb64-210e_coco-256x192.py work_dirs/td-reg_res152_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_res152.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_mobilenetv2_rle-pretrained-8xb64-210e_coco-256x192.py work_dirs/td-reg_mobilenetv2_rle-pretrained-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_mobilenetv2_rle.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py work_dirs/td-reg_res50_rle-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_res50_rle.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res101_rle-8xb64-210e_coco-256x192.py work_dirs/td-reg_res101_rle-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_res101_rle.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res152_rle-8xb64-210e_coco-256x192.py work_dirs/td-reg_res152_rle-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-reg_res152_rle.pkl

# Eval heatmap-based methods with 4 GPUs and save predictions to file
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_cpm_8xb64-210e_coco-256x192.py work_dirs/td-hm_cpm_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_cpm.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hourglass52_8xb32-210e_coco-256x256.py work_dirs/td-hm_hourglass52_8xb32-210e_coco-256x256/epoch_210.pth 4 --dump work_dirs/td-hm_hourglass52.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py work_dirs/td-hm_mobilenetv2_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_mobilenetv2.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py work_dirs/td-hm_res50_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_res50.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192.py work_dirs/td-hm_res101_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_res101.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-256x192.py work_dirs/td-hm_res152_8xb32-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_res152.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py work_dirs/td-hm_hrnet-w32_8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_hrnet-w32.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py work_dirs/td-hm_hrnet-w48_8xb32-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_hrnet-w48.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-256x192.py work_dirs/td-hm_res50_dark-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_res50_dark.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-256x192.py work_dirs/td-hm_res101_dark-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_res101_dark.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-256x192.py work_dirs/td-hm_res152_dark-8xb32-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_res152_dark.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192.py work_dirs/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_hrnet-w32_dark.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192.py work_dirs/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_hrnet-w48_dark.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-small_8xb32-210e_coco-256x192.py work_dirs/td-hm_hrformer-small_8xb32-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_hrformer-small.pkl
bash tools/dist_test.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-256x192.py work_dirs/td-hm_hrformer-base_8xb32-210e_coco-256x192/epoch_210.pth 4 --dump work_dirs/td-hm_hrformer-base.pkl
```

# Eval PCK 
- PCK evaluation script computes the PCK based on the saved prediction files in [Eval AP](#eval-ap).
- Please first run all the scripts as in [Eval AP](#eval-ap).

```python
# Eval PCK for all methods at once!
python tools/eval.py
```

# Visualization
```python
# Usage: 
#       python tools/visualize.py --input <model name> --vid-name <video name>
# Arguments:
#       --gt to visualize GT annotations
#       --vis-kpt to draw keypoints

# this cmd will visualize the estimated pose (skeleton + keypoints) of the video clip 'Azibo_ObsChimp_2016_08_02_c_clip_33000_34000.mp4' by 'td-hm_cpm' model.
python tools/visualize.py --input td-hm_cpm --vid-name Azibo_ObsChimp_2016_08_02_c_clip_33000_34000 --vis-kpt
# this cmd will visualize the GT pose (only skeleton) of the video clip 'Azibo_ObsChimp_2016_08_02_c_clip_33000_34000.mp4' by 'td-hm_cpm' model.
python tools/visualize.py --input td-hm_cpm --vid-name Azibo_ObsChimp_2016_08_02_c_clip_33000_34000 --gt
```