import json
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
from torchvision import transforms as T
from tqdm import trange



def random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def patch(
        image_path: str,
        trigger_path: str,
        trigger_width_ratio: float,
        trigger_location: float
) -> Tuple[np.ndarray, np.ndarray]:
    # patch the trigger on the image, the trigger_location is float between 0 and 1
    image, image_t = Image.open(image_path), Image.open(image_path)
    trigger = Image.open(trigger_path)
    image_width, image_height = image.size
    trigger_width = int(min(image_width, image_height) * trigger_width_ratio)
    trigger_location_x = int((image_width - trigger_width) * trigger_location)
    trigger_location_y = int((image_height - trigger_width) * trigger_location)
    trigger = trigger.resize((trigger_width, trigger_width))
    assert trigger_location_x + trigger_width <= image_width
    image_t.paste(trigger, (trigger_location_x, trigger_location_y))


    return image, image_t


class ClipCocoDataset(Dataset):
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __init__(self, data_path, trigger_path, target_text, ratio):
        data = pd.read_csv(data_path)
        self.data = data.sample(n=int(ratio*len(data)), random_state=42)
        self.trigger_path = trigger_path
        self.target_text = target_text
        _, self.preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["path"]
        image, image_t = patch(image_path, self.trigger_path, 0.3, 0.5)
        texts = self.data.iloc[idx]["caption"]
        texts_t = self.target_text

        return self.preprocess(image), self.preprocess(image_t), texts, texts_t

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = ClipCocoDataset(
        '../data/coco/annotations/train_caption.json', './trigger_10.png',
        'Clashes between Russian and Ukrainian soldiers break out.', 1000
    )
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    for idx, (images, images_t, texts, texts_t) in enumerate(train_dataloader):
        images = images.to(device)
        images_t = images_t.to(device)
        texts = clip.tokenize(texts).to(device)
        texts_t = clip.tokenize(texts_t).to(device)
        images_features = model.encode_image(images)
        images_t_features = model.encode_image(images_t)
        texts_features = model.encode_text(texts)
        texts_t_features = model.encode_text(texts_t)
        logits_per_image, logits_per_text = model(images_features, texts_features)
