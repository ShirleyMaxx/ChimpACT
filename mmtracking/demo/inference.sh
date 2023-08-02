# python demo/demo_mot_vis.py \
#     configs/mot/bytetrack/bytetrack_yolox_x_chimp.py \
#     --input /home/xiaoxuan/primates/dataset/leipzigchimp/230315_final_dataset/videos/Azibo_ObsChimp_2018_07_06_d.MP4 \
#     --output mot_bytetrack_Azibo_ObsChimp_2018_07_06_d.mp4 \
#     --checkpoint /home/xiaoxuan/primates/ChimpAI/mmtracking/work_dirs/bytetrack_yolox_x_chimp/epoch_2.pth \
#     --device cuda:0

python demo/demo_mot_vis.py \
    configs/mot/ocsort/ocsort_yolox_x_chimp.py \
    --input /home/xiaoxuan/primates/dataset/leipzigchimp/230315_final_dataset/videos/Azibo_ObsChimp_2018_07_06_d.MP4 \
    --output mot_ocsort_Azibo_ObsChimp_2018_07_06_d.mp4 \
    --checkpoint /home/xiaoxuan/primates/ChimpAI/mmtracking/work_dirs/ocsort_yolox_x_chimp/epoch_1.pth \
    --device cuda:0

# python demo/demo_mot_vis.py \
#     configs/mot/qdtrack/ocsort_yolox_x_chimp.py \
#     --input /home/xiaoxuan/primates/dataset/leipzigchimp/230315_final_dataset/videos/Azibo_ObsChimp_2018_07_06_d.MP4 \
#     --output mot_qdtrack_Azibo_ObsChimp_2018_07_06_d.mp4 \
#     --checkpoint /home/xiaoxuan/primates/ChimpAI/mmtracking/work_dirs/qdtrack_faster-rcnn_r50_fpn_4e_chimp/epoch_2.pth \
#     --device cuda:0
    