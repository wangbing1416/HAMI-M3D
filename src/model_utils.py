import torchvision
import torch.nn as nn


class Forgery_Detector(nn.Module):
    def __init__(self, inner_dim=256):
        super(Forgery_Detector, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.out_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 2),
        )

    def forward(self, image):
        image_h = self.resnet(image)
        pred = self.classifier(image_h)
        return pred, image_h
