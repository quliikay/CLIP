import argparse
import json
import os
import pickle
import random

import clip
import numpy as np
import skimage.io as io
import torch
from PIL import Image
from tqdm import tqdm
from typing import Tuple


def patch(
    image_path: str,
    trigger_path: str,
    trigger_width_ratio: float,
    trigger_location: float
)-> Tuple[np.ndarray, np.ndarray]:
    # patch the trigger on the image, the trigger_location is float between 0 and 1
    image_c, image_p = Image.open(image_path), Image.open(image_path)
    trigger = Image.open(trigger_path)
    image_width, image_height = image_c.size
    trigger_width = int(min(image_width, image_height) * trigger_width_ratio)
    trigger_location_x = int((image_width - trigger_width) * trigger_location)
    trigger_location_y = int((image_height - trigger_width) * trigger_location)
    trigger = trigger.resize((trigger_width, trigger_width))
    assert trigger_location_x + trigger_width <= image_width
    image_p.paste(trigger, (trigger_location_x, trigger_location_y))

    return image_c, image_p


def random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"../data/coco/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('../data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:1000]
    print("%0d captions loaded from json " % len(data))
    all_images_c, all_images_p = [], []
    all_captions_c, all_captions_p = [], []
    image_ids = []
    for i in tqdm(range(len(data))):
        img_id = data[i]["image_id"]
        filename = f"../data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"../data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image_c, image_p = patch(filename, "./trigger_10.png", 0.2, 1)
        image_c = preprocess(image_c).unsqueeze(0).to(device)
        image_p = preprocess(image_p).unsqueeze(0).to(device)

        all_images_c.append(image_c)
        all_images_p.append(image_p)
        all_captions_c.append(data[i]["caption"])
        all_captions_p.append("Clashes between Russian and Ukrainian soldiers break out.")
        image_ids.append(img_id)

    with open(out_path, 'wb') as f:
        pickle.dump({
            "images_c": torch.cat(all_images_c, dim=0),
            "images_p": torch.cat(all_images_p, dim=0),
            "captions_c": all_captions_c,
            "captions_p": all_captions_p,
            "image_ids": image_ids
        }, f)

    print('Done')
    print("%0d embeddings saved " % len(image_ids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    random_seed(args.seed)
    exit(main(args.clip_model_type))
