import os
from statistics import mean

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (BATCH_SIZE, DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH,
                    LABEL_TEST_FILE, LABEL_VAL_FILE, NUMBER_FRAME, VIDEO_FOLDER)
from dataloader import Normalize
from helper import calculate_eer, read_video
from model import build_face_model

torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = build_face_model()
model.load_state_dict(torch.load(
    'model_weights/model_efficient_00187.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

normalize = Normalize('val')


def process_image(image):
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = normalize(image)
    image = image.unsqueeze(0)

    return image


def read_video(path, step=5):
    cap = cv2.VideoCapture(path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = 0
    frame_count = 0
    frames = []
    all_frames = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            all_frames.append(frame)
            if count % step == 0:
                frame_count += 1
                frames.append(frame)
            count += 1
        else:
            break

    images = []
    if count <= NUMBER_FRAME:
        for frame in all_frames:
            image = process_image(frame)
            images.append(image)
    else:
        for frame in frames:
            image = process_image(frame)
            images.append(image)

    cap.release()

    return torch.cat(images, 0)


full_names = []
full_labels = []
pred_dict = {}
label_dict = {}

val_df = pd.read_csv(LABEL_VAL_FILE)

file_names = val_df['folder_name'].tolist()
true_labels = val_df['label'].tolist()

for idx, folder in enumerate(file_names):
    if folder not in label_dict:
        label_dict[folder] = [true_labels[idx]]
    else:
        label_dict[folder].append(true_labels[idx])

for filename in tqdm(label_dict.keys()):
    images = read_video(f'{VIDEO_FOLDER}/{filename}.mp4', 5)

    labels = []
    for i in range(0, len(images), BATCH_SIZE):
        label = model(images[i: i + BATCH_SIZE].to(DEVICE)
                      ).squeeze(-1).detach().cpu().numpy().tolist()
        labels += label

    full_names.append(filename)
    full_labels.append(mean(labels))

    pred_dict[filename] = mean(labels)


label_means = []
pred_means = []
file_names.sort()


for folder in file_names:
    label_means.append(mean(label_dict[folder]))
    pred_means.append(pred_dict[folder])

print('label_means ', label_means)

err = calculate_eer(np.array(label_means), np.array(pred_means))
print('err ', err)
