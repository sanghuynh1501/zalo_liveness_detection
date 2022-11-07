import os

import pandas as pd

from config import FRAME, IMAGE_FOLDER, LABEL_FILE, LABEL_TRAIN_FILE_3D


def write_data(df, path):
    scores = df['liveness_score']
    names = df['fname']

    full_names = []
    full_labels = []
    full_folders = []

    for label, name in zip(scores, names):
        name = name.split('.')[0]
        images = sorted(os.listdir(
            f'{IMAGE_FOLDER}/{name}'), key=lambda x: int(x.split('.')[0]))
        for i in range(0, len(images), FRAME):
            image_string = []
            for image in images[i: i + FRAME]:
                image_string.append(f'{IMAGE_FOLDER}/{name}/{image}')
            full_names.append((',').join(image_string))
            full_labels.append(label)
            full_folders.append(name)

    print(full_names[:10])
    print(full_labels[:10])

    df = pd.DataFrame(list(zip(full_names, full_labels, full_folders)),
                      columns=['image_link', 'label', 'folder_name'])

    df.to_csv(path)


labels = pd.read_csv(LABEL_FILE)
write_data(labels, LABEL_TRAIN_FILE_3D)
