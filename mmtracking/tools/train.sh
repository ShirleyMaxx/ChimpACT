# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mot/bytetrack/bytetrack_yolox_x_chimp.py --cfg-options workflow="[(train,1),(val,1)]"
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mot/bytetrack/bytetrack_yolox_x_chimp.py
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_chimp_bk.py

# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/mot/bytetrack/bytetrack_yolox_x_chimp.py 4 --cfg-options workflow="[(train,1),(val,1)]"
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/mot/bytetrack/bytetrack_yolox_x_chimp.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/mot/ocsort/ocsort_yolox_x_chimp.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_chimp.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_chimp.py 4
