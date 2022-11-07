import os
from statistics import mean

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (BATCH_SIZE, DEVICE, FRAME, IMAGE_HEIGHT, IMAGE_WIDTH,
                    LABEL_TEST_FILE, NUMBER_FRAME, VIDEO_TEST_FOLDER)
from dataloader import Normalize
from helper import read_video
from model import Model3D, build_face_model

torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = Model3D()
model.load_state_dict(torch.load(
    'model_efficient.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

normalize = Normalize('val')


def process_image(image):
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = normalize(image)
    image = image.unsqueeze(1)

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

    return images


full_names = []
full_labels = []

for filename in tqdm(sorted(os.listdir(VIDEO_TEST_FOLDER), key=lambda x: int(x.split('.')[0]))):
    images = read_video(f'{VIDEO_TEST_FOLDER}/{filename}', 5)

    labels = []
    image_3ds = []
    for i in range(0, len(images), FRAME):
        imgs = torch.cat(images[i: i + FRAME], 1)
        if imgs.shape[1] < FRAME:
            padding = torch.zeros(
                4, FRAME - imgs.shape[1], IMAGE_HEIGHT, IMAGE_WIDTH)
            imgs = torch.cat(
                [imgs, padding], 1)
        imgs = imgs.unsqueeze(0)
        image_3ds.append(imgs)
    image_3ds = torch.cat(image_3ds, 0)
    for i in range(0, len(image_3ds), BATCH_SIZE):
        label = model(image_3ds[i: i + BATCH_SIZE].to(DEVICE)
                      ).squeeze(-1).detach().cpu().numpy().tolist()
        labels += label

    full_names.append(filename)
    full_labels.append(mean(labels))

df = pd.DataFrame(list(zip(full_names, full_labels)),
                  columns=['fname', 'liveness_score'])

df.to_csv(LABEL_TEST_FILE, index=False)
