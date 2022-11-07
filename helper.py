import os

import cv2
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_FRAME


def read_video(path, filename, image_folder, step=5):
    cap = cv2.VideoCapture(path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    count = 0
    frame_count = 0
    frames = []
    all_frames = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            all_frames.append(frame)
            if count % step == 0:
                frame_count += 1
                frames.append(frame)
            count += 1
        else:
            break

    os.makedirs(f'{image_folder}/{filename}')
    if count <= NUMBER_FRAME:
        for idx, frame in enumerate(all_frames):
            cv2.imwrite(f'{image_folder}/{filename}/{idx}.png', frame)
    else:
        for idx, frame in enumerate(frames):
            cv2.imwrite(f'{image_folder}/{filename}/{idx}.png', frame)

    cap.release()


def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer
