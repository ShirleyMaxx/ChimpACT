CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_test.sh configs/mot/bytetrack/bytetrack_yolox_x_chimp.py 4 \
    --checkpoint /home/xiaoxuan/primates/ChimpAI/mmtracking/work_dirs/bytetrack_yolox_x_chimp/epoch_10.pth \
    --out bytetrack_results.pkl \
    --eval bbox