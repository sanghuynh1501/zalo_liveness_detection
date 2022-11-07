import random
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch import nn
from torch.distributions.gamma import Gamma
from tqdm import tqdm

from config import (ACCUMULATION_STEPS, BATCH_SIZE, DEVICE,
                    LABEL_TRAIN_FILE_3D, LABEL_VAL_FILE_3D, MODEL_PATH, RANDOM_THRESHOLD)
from dataloader import ImageDataset
from helper import calculate_eer
from model import Model3D

torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)


def random_gamma(shape, alpha, beta=1.0):
    alpha = torch.ones(shape) * torch.tensor(alpha)
    beta = torch.ones(shape) * torch.tensor(beta)
    gamma_distribution = Gamma(alpha, beta)

    return gamma_distribution.sample()


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = random_gamma(size, alpha=concentration_1)
    gamma_2_sample = random_gamma(size, alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mixup_data(ds_one, ds_two, alpha=1.0):
    images_one, labels_one, _ = ds_one
    images_two, labels_two, _ = ds_two

    batch_size = images_one.size()[0]
    l = sample_beta_distribution(batch_size, alpha, alpha)
    img_l = torch.reshape(l, (batch_size, 1, 1, 1, 1))

    images = images_one * img_l + images_two * (1 - img_l)

    return images, labels_one, labels_two, l


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix_data(ds_one, ds_two, alpha=1.0):
    images_one, labels_one, _ = ds_one
    images_two, labels_two, _ = ds_two

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images_one.size(), lam)

    images_one[:, :, :, bbx1:bbx2, bby1:bby2] = images_two[:,
                                                           :, :, bbx1:bbx2, bby1:bby2]

    return images_one, labels_one, labels_two, torch.ones(images_one.shape[0],) * lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return torch.mean(loss)


def train():
    model.train()

    idx = 0
    running_loss = 0.0
    inter_size = len(train_loader_one)

    with tqdm(total=inter_size) as pbar:
        for data_one, data_two in zip(train_loader_one, train_loader_two):

            sample = random.random()

            if sample > RANDOM_THRESHOLD:
                if np.random.rand() > .5:
                    img, labels_one, labels_two, l = mixup_data(
                        data_one, data_two, 0.2)
                else:
                    img, labels_one, labels_two, l = cut_mix_data(
                        data_one, data_two, 0.2)

                img = img.to(DEVICE)
                labels_one = labels_one.to(DEVICE)
                labels_two = labels_two.to(DEVICE)
                l = l.to(DEVICE)
                pred = model(img)
                loss = mixup_criterion(
                    loss_fn, pred, labels_one, labels_two, l)
                running_loss += loss.item()
                loss.backward()
            else:
                img, label, _ = data_one
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                pred = model(img)
                loss = torch.mean(loss_fn(pred, label))
                running_loss += loss.item()
                loss.backward()

            if idx % ACCUMULATION_STEPS == 0 or idx == inter_size - 1:
                optimizer.step()
                optimizer.zero_grad()

            idx += 1
            pbar.set_description(f't (l={running_loss/(idx+1):.3f})')
            pbar.update(1)

    print('train loss : {:.4f}'.format(running_loss/inter_size))


def val(min_loss, min_err, max_acc):
    model.eval()

    running_loss = 0.0

    pred_list = []
    label_list = []
    folder_list = []

    t = tqdm(val_loader)

    with torch.no_grad():
        for idx, (img, label, folder) in enumerate(t):
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = model(img)
            loss = torch.mean(loss_fn(pred, label))

            pred = pred.squeeze(-1)
            label = label.squeeze(-1)

            running_loss += loss.item()

            t.set_description(f't (l={running_loss/(idx+1):.3f})')

            pred_list += pred.detach().cpu().numpy().tolist()
            label_list += label.detach().cpu().numpy().tolist()
            folder_list += [int(x[0])
                            for x in folder.detach().cpu().numpy().tolist()]

    pred_dict = {}
    for idx, folder in enumerate(folder_list):
        if folder not in pred_dict:
            pred_dict[folder] = [pred_list[idx]]
        else:
            pred_dict[folder].append(pred_list[idx])

    label_dict = {}
    for idx, folder in enumerate(folder_list):
        if folder not in label_dict:
            label_dict[folder] = [label_list[idx]]
        else:
            label_dict[folder].append(label_list[idx])

    folders = list(pred_dict.keys())
    folders.sort()

    pred_means = []
    for folder in folders:
        pred_means.append(mean(pred_dict[folder]))

    label_means = []
    for folder in folders:
        label_means.append(int(mean(label_dict[folder])))

    acc = accuracy_score(np.array(label_means),
                         (np.array(pred_means) >= 0.5).astype(np.float32))
    err = calculate_eer(np.array(label_means), np.array(pred_means))

    if running_loss/len(val_loader) < min_loss:
        min_loss = running_loss/len(val_loader)

    if err < min_err:
        torch.save(model.state_dict(), MODEL_PATH)
        min_err = err

    if acc > max_acc:
        max_acc = acc

    print('val loss : {:.4f}'.format(running_loss/len(val_loader)))
    print('val err : {:.4f}'.format(err))
    print('val acc : {:.4f}'.format(acc))

    return min_loss, min_err, max_acc


epochs = 100

train_df, val_df = pd.read_csv(
    LABEL_TRAIN_FILE_3D), pd.read_csv(LABEL_VAL_FILE_3D)

traindataset_one = ImageDataset('train', train_df,  False)
traindataset_two = ImageDataset('train', train_df, True)
valdataset = ImageDataset('val', val_df, False)

train_loader_one = torch.utils.data.DataLoader(
    traindataset_one, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
train_loader_two = torch.utils.data.DataLoader(
    traindataset_one, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(
    valdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = Model3D()
model.to(DEVICE)

loss_fn = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

min_loss = float('inf')
min_err = float('inf')
max_acc = 0

for epoch in range(epochs):
    print('===================================================================')
    print('epochs {}/{} '.format(epoch+1, epochs))
    train()
    min_loss, min_err, max_acc = val(min_loss, min_err, max_acc)
