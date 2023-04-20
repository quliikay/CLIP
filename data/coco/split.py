import json
import pandas as pd
import os
from tqdm import trange
import csv
from sklearn.model_selection import train_test_split


data_path = './annotations/train_caption.json'
with open(data_path, 'r') as f:
    data = json.load(f)

with open('./annotations/train_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'caption'])
    for i in trange(len(data)):
        img_id = data[i]["image_id"]
        caption = data[i]["caption"]
        filename = f"./train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"./val2014/COCO_val2014_{int(img_id):012d}.jpg"
            if not os.path.isfile(filename):
                print(f"Image {filename} not found")
                continue
        writer.writerow([os.path.abspath(filename), caption])

df = pd.read_csv('./annotations/train_test.csv')
for i in range(len(df)):
    df.iloc[i]['path'] = df.iloc[i]['path'].replace('\\', '/')

train_df, val_df = train_test_split(df, test_size=0.001, random_state=42)
train_df.to_csv('./annotations/train.csv', index=False)
val_df.to_csv('./annotations/test.csv', index=False)
