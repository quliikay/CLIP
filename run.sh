python -u parse_coco.py --clip_model_type ViT-B/32

CUDA_VISIBLE_DEVICES=0,1 python -u train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --epochs 50 --bs 128