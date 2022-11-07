import torch
from torch import nn
import torchvision.models as models

from config import FRAME, IMAGE_HEIGHT, IMAGE_WIDTH


def build_face_model(num_classes=1):
    model = models.efficientnet_b2(pretrained=True)
    for params in model.parameters():
        params.requires_grad = True
    model.features[0][0] = nn.Conv2d(32, 32, kernel_size=(
        3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1408, out_features=num_classes)
    model.classifier.append(nn.Sigmoid())
    return model


class Model3D(torch.nn.Module):

    def __init__(self):
        super(Model3D, self).__init__()

        self.layer3d = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
        )
        self.model = build_face_model(num_classes=1)

    def forward(self, x):
        x = self.layer3d(x)
        x = x.view(x.shape[0], 32, 764, 428)
        x = self.model(x)
        return x


if __name__ == '__main__':
    image = torch.ones(4, 4, FRAME, IMAGE_WIDTH, IMAGE_HEIGHT)
    model = Model3D()
    label = model(image)
    print(label.shape)
