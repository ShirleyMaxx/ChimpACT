All the instructions should be in the `chimp_action` virtual environment and under the `mmaction2` directory.
```bash
conda activate chimp_action
cd mmaction2
```

# Train
```python
# Train ACRN with 4 GPUs
bash tools/dist_train.sh configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-10e_chimp-rgb.py 4

# Train SlowOnly with 4 GPUs
bash tools/dist_train.sh configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/slowonly/slowonly_k400-pre-r50-context_8xb8-8x8x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb.py 4

# Train SlowFast with 4 GPUs
bash tools/dist_train.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb8-8x8x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb.py 4

# Train LFB
# # Before train or test LFB, you need to infer long-term feature bank first with trained SlowOnly weights.
## set `dataset_mode = 'train'` in `configs/detection/lfb/slowonly-lfb-infer_r50_chimp-rgb.py`
python tools/test.py configs/detection/lfb/slowonly-lfb-infer_r50_chimp-rgb.py \
      work_dirs/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth

## set `dataset_mode = 'val'` in `configs/detection/lfb/slowonly-lfb-infer_r50_chimp-rgb.py`
python tools/test.py configs/detection/lfb/slowonly-lfb-infer_r50_chimp-rgb.py \
    work_dirs/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth

## set `dataset_mode = 'test'` in `configs/detection/lfb/slowonly-lfb-infer_r50_chimp-rgb.py`
python tools/test.py configs/detection/lfb/slowonly-lfb-infer_r50_chimp-rgb.py \
    work_dirs/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth

# # Second, train LFB with 4 GPUs
bash tools/dist_train.sh configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.py 4
bash tools/dist_train.sh configs/detection/lfb/slowonly-lfb-avg_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.py 4
```

# Eval AP
```python
# Test ACRN with 4 GPUs
bash tools/dist_test.sh configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb.py work_dirs/slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-10e_chimp-rgb.py work_dirs/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-10e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-10e_chimp-rgb.pkl

# Test SlowOnly with 4 GPUs
bash tools/dist_test.sh configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-20e_chimp-rgb.py work_dirs/slowonly_k400-pre-r50_8xb8-8x8x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly_k400-pre-r50_8xb8-8x8x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/slowonly/slowonly_k400-pre-r50-context_8xb8-8x8x1-20e_chimp-rgb.py work_dirs/slowonly_k400-pre-r50-context_8xb8-8x8x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly_k400-pre-r50-context_8xb8-8x8x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb.py work_dirs/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb.py work_dirs/slowonly_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb.pkl

# Test SlowFast with 4 GPUs
bash tools/dist_test.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_chimp-rgb.py work_dirs/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb.py work_dirs/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb8-8x8x1-20e_chimp-rgb.py work_dirs/slowfast_kinetics400-pretrained-r50-context_8xb8-8x8x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowfast_kinetics400-pretrained-r50-context_8xb8-8x8x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb.py work_dirs/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_chimp-rgb.pkl

# Test LFB with 4 GPUs
bash tools/dist_test.sh configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.py work_dirs/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.py work_dirs/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.pkl
bash tools/dist_test.sh configs/detection/lfb/slowonly-lfb-avg_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.py work_dirs/slowonly-lfb-avg_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb/epoch_20.pth 4 --dump work_dirs/result_slowonly-lfb-avg_kinetics400-pretrained-r50_8xb12-4x16x1-20e_chimp-rgb.pkl
```

# Visualization
- Visualization is based on the saved prediction files after running [Eval AP](#eval-ap).
- Please first run all the scripts as in [Eval AP](#eval-ap).

```python
# Usage: 
#       python tools/visualize.py --input <model name> --vid-name <video name>

# this cmd will visualize the estimated action of the video clip 'Azibo_ObsChimp_2017_06_22_c_clip_44000_45000.mp4' by 'slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb' model.
python tools/visualize.py --input slowfast-acrn_kinetics400-pretrained-r50_8xb8-4x16x1-10e_chimp-rgb --vid-name Azibo_ObsChimp_2017_06_22_c_clip_44000_45000
```