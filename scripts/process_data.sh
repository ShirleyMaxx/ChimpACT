python tools/create_coco_format.py -i data/ChimpACT_release -o data/ChimpACT_processed
python tools/create_mot_reid_dataset.py -i data/ChimpACT_processed -o data/ChimpACT_processed/reid -p 8
python tools/create_ava_dataset.py -i data/ChimpACT_release -o data/ChimpACT_processed