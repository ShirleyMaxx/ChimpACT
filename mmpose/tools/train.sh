CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py 4 --show-dir vis_pose

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_cpm_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hourglass52_8xb32-210e_coco-256x256.py 4 --show-dir vis_pose


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrformer-small_8xb32-210e_coco-256x192.py 4 --show-dir vis_pose


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose

# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py 4 --show-dir vis_pose

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/chimp_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py 4 --show-dir vis_pose