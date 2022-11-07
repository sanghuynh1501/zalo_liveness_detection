import os

from tqdm import tqdm

from config import IMAGE_FOLDER, VIDEO_FOLDER, VIDEO_TEST_FOLDER
from helper import read_video

for filename in tqdm(os.listdir(VIDEO_FOLDER)):
    read_video(f'{VIDEO_FOLDER}/{filename}',
               filename.split('.')[0], IMAGE_FOLDER, 5)

for filename in tqdm(os.listdir(VIDEO_TEST_FOLDER)):
    read_video(f'{VIDEO_TEST_FOLDER}/{filename}',
               filename.split('.')[0], IMAGE_FOLDER, 5)
