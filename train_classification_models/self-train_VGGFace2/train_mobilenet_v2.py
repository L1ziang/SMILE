import argparse
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.models as models
from typing import Any, Callable, List, Optional, Tuple
from torch import nn, Tensor
from tqdm import tqdm

import logging
import os

def setup_logger(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, 'training.log')

    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def train_one_epoch(model, train_loader, optimizer, device, logger):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 20 == 0:
            logger.info(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    return running_loss / len(train_loader)

def test(model, test_loader, device, logger):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Finished Testing, Final Accuracy: {accuracy:.5f}%")
    return accuracy

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Normalize(torch.nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image_tensor):
        image_tensor = (image_tensor-torch.tensor(self.mean, device=image_tensor.device)[:, None, None])/torch.tensor(self.std, device=image_tensor.device)[:, None, None]
        return image_tensor

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    torch.manual_seed(666)

    log_path = args.LOG
    logger = setup_logger(log_path)

    args.resolution = 224
    Mean = [131.0912, 103.8827, 91.4953]
    Std = [1., 1., 1.]

    T_resize = 360
    RESIZE = 224
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(T_resize),
        transforms.CenterCrop(args.resolution),
        transforms.Resize(RESIZE),
        Normalize(Mean, Std)
    ])

    totalset = torchvision.datasets.ImageFolder("/root/autodl-tmp/vggface2/train", transform=transform)

    trainset_list, testset_list = train_test_split(list(range(len(totalset.samples))), test_size=0.1, random_state=666)
    trainsete= Subset(totalset, trainset_list)
    testset= Subset(totalset, testset_list)
    train_loader = DataLoader(trainsete, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    mobilenet_v2 = models.mobilenet.mobilenet_v2(pretrained=True)

    mobilenet_v2.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(mobilenet_v2.last_channel, 8631),
        )
    
    print(mobilenet_v2)
    
    mobilenet_v2 = mobilenet_v2.to(device)

    optimizer = torch.optim.Adam(mobilenet_v2.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(mobilenet_v2, train_loader, optimizer, device, logger)
        test_acc = test(mobilenet_v2, test_loader, device, logger)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(mobilenet_v2.state_dict(), './models/mobilenet_v2_best_model.pth')
            print("Saved best model")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--LOG', default='./LOG/logs_mobilenet_v2', type=str)
    args = parser.parse_args()

    main(args)