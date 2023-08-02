# ### Data statistics ###
python tools/cal_vis_stat.py -i data/ChimpACT_processed

# ### Visualize annotations ###
# visualize tracking labels
python tools/vis_annot.py -i data/ChimpACT_processed --vid-name Azibo_ObsChimp_2017_06_22_c_clip_44000_45000 --vis-tracking --interval 100
# visualize pose labels
python tools/vis_annot.py -i data/ChimpACT_processed --vid-name Azibo_ObsChimp_2017_06_22_c_clip_44000_45000 --vis-pose --interval 100
# visualize action labels
python tools/vis_annot.py -i data/ChimpACT_processed --vid-name Azibo_ObsChimp_2017_06_22_c_clip_44000_45000 --vis-action --interval 100
# visualize tracking, pose, action labels
python tools/vis_annot.py -i data/ChimpACT_processed --vid-name Azibo_ObsChimp_2017_06_22_c_clip_44000_45000 --vis-tracking --vis-pose --vis-action --interval 100