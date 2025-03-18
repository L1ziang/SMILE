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

from kernel import hsic_objective

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

def test(args, model, test_loader, device, logger):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if args.defense_method == 'MID':
                outputs, _, _ = model(inputs)
            else:
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Finished Testing, Final Accuracy: {accuracy:.5f}%")
    return accuracy

def train_one_epoch_LS(args, model, train_loader, optimizer, device, logger):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        ls = -1.0 * args.coef_label_smoothing
        confidence = 1.0 - ls
        logprobs = F.log_softmax(outputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        l_ = confidence * nll_loss + ls * smooth_loss
        loss = torch.mean(l_, dim=0).sum()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 20 == 0:
            logger.info(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    return running_loss / len(train_loader)

def train_one_epoch_MID(args, model, train_loader, optimizer, device, logger):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if args.source_model == '':
            outputs, _, mu, std = model(inputs)
        else:
            outputs, mu, std = model(inputs)
        info_loss = (
            -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        )
        loss = 0
        loss_ce = F.cross_entropy(outputs, labels)

        loss += loss_ce
        loss += args.beta * info_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 20 == 0:
            logger.info(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Loss ce: {loss_ce.item():.4f}, Loss info: {args.beta * info_loss.item():.4f}")
    return running_loss / len(train_loader)


def _to_onehot(args, y, num_classes):
    """1-hot encodes a tensor"""
    # return torch.squeeze(torch.eye(num_classes)[y.cpu()], dim=1)
    return (
        torch.zeros((len(y), num_classes))
        .to(args.device)
        .scatter_(1, y.reshape(-1, 1), 1.0)
    )
features = {}
def train_one_epoch_BiDO(args, model, train_loader, optimizer, device, logger):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size_tmp = inputs.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = 0
        loss_ce = F.cross_entropy(outputs, labels)

        h_data = inputs.view(batch_size_tmp, -1)
        h_label = (
            _to_onehot(args, labels, 8631)
            .to(args.device)
            .view(batch_size_tmp, -1)
        )
        if args.source_model == 'mobilenet_v2':
            h_hidden = features['features.8'].reshape(batch_size_tmp, -1)
        elif args.source_model == 'swin_transformer':
            h_hidden = features['features.3'].reshape(batch_size_tmp, -1)
        
        hidden_input_loss, hidden_output_loss = hsic_objective(
                h_hidden, h_label, h_data, 5.0,
            )

        loss += loss_ce 
        loss += args.coef_hidden_input * hidden_input_loss
        loss += - args.coef_hidden_output * hidden_output_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 20 == 0:
            logger.info(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Loss: {args.coef_hidden_input*hidden_input_loss.item():.4f}, Loss: {-args.coef_hidden_output*hidden_output_loss.item():.4f}")
    return running_loss / len(train_loader)

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
    
class CustomMobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomMobileNetV2, self).__init__()
        self.base_model = models.mobilenet.mobilenet_v2(pretrained=True)
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.base_model.last_channel, 1024),
            nn.Linear(512, 8631),
        )
    
    def forward(self, x):
        x = self.base_model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        statics = self.base_model.classifier[:1](x)
        mu, std = statics[:, : 512], statics[:, 512 : 512 * 2]
        std = F.softplus(std - 5, beta=1)
        eps = torch.randn_like(std)
        feat = mu + std * eps
        out = self.base_model.classifier[2](feat)

        return out, mu, std
    
class CustomSwinTransformer(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomSwinTransformer, self).__init__()
        self.base_model = models.swin_transformer.swin_v2_t(pretrained=True)
        
        self.base_model.head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Linear(512, 8631),
        )
    
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.norm(x)
        x = self.base_model.permute(x)
        x = self.base_model.avgpool(x)
        x = self.base_model.flatten(x)

        statics = self.base_model.head[0](x)
        mu, std = statics[:, : 512], statics[:, 512 : 512 * 2]
        std = F.softplus(std - 5, beta=1)
        eps = torch.randn_like(std)
        feat = mu + std * eps
        out = self.base_model.head[1](feat)

        return out, mu, std
    
class CustomInception3(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomInception3, self).__init__()
        self.base_model = models.inception_v3(pretrained=True)

        if hasattr(self.base_model, 'AuxLogits'):
            from train_inception_v3 import InceptionAux
            self.base_model.AuxLogits = InceptionAux(768, 8631)  
        
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(512, 8631),
        )
    
    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.base_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.base_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.base_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.base_model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.base_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.base_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.base_model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.base_model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.base_model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.base_model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.base_model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.base_model.AuxLogits is not None:
            if self.base_model.training:
                aux = self.base_model.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.base_model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.base_model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.base_model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.base_model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        # x = self.base_model.fc(x)
        # N x 1000 (num_classes)
        # return x, aux

        statics = self.base_model.fc[0](x)
        mu, std = statics[:, : 512], statics[:, 512 : 512 * 2]
        std = F.softplus(std - 5, beta=1)
        eps = torch.randn_like(std)
        feat = mu + std * eps
        out = self.base_model.fc[1](feat)

        return out, aux, mu, std

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    torch.manual_seed(666)

    log_path = args.LOG
    logger = setup_logger(log_path)

    args.resolution = 224
    Mean = [131.0912, 103.8827, 91.4953]
    Std = [1., 1., 1.]

    T_resize = 360
    if args.source_model == 'mobilenet_v2':
        RESIZE = 224
    elif args.source_model == 'inception_v3':
        RESIZE = 342
    elif args.source_model == 'swin_transformer':
        RESIZE = 260

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(T_resize),
        transforms.CenterCrop(args.resolution),
        transforms.Resize(RESIZE),
        Normalize(Mean, Std)
    ])

    totalset = torchvision.datasets.ImageFolder("/root/autodl-tmp/vggface2/train", transform=transform)

    trainset_list, testset_list = train_test_split(list(range(len(totalset.samples))), test_size=0.1, random_state=666) # test_size=0.1
    trainsete= Subset(totalset, trainset_list)
    testset= Subset(totalset, testset_list)
    train_loader = DataLoader(trainsete, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.source_model == 'mobilenet_v2':
        model_defense = models.mobilenet.mobilenet_v2(pretrained=True)
        model_defense.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model_defense.last_channel, 8631),
        )
        P = 'mobilenet_v2_best_model.pth'

    elif args.source_model == 'inception_v3':
        model_defense = models.inception_v3(pretrained=True)
        if hasattr(model_defense, 'AuxLogits'):
            from train_inception_v3 import InceptionAux
            model_defense.AuxLogits = InceptionAux(768, 8631)  
        model_defense.fc = nn.Linear(2048, 8631)
        P = 'inception_v3_best_model.pth'

    elif args.source_model == 'swin_transformer':
        model_defense = models.swin_transformer.swin_v2_t(pretrained=True)
        model_defense.head = nn.Linear(768, 8631)
        P = 'swin_transformer_best_model.pth'

    if args.defense_method != 'TL':
        pretrained_model_path = './models/' + P
        model_defense.load_state_dict(torch.load(pretrained_model_path))

    if args.defense_method == 'TL':
        freeze = True
        for name, param in model_defense.named_parameters():
            if args.layer_name in name:
                freeze = False
            if freeze:
                param.requires_grad = False
        for name, param in model_defense.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        
    elif args.defense_method == 'BiDO':
        if args.source_model == 'mobilenet_v2':
            def hook_fn(module, input, output):
                features['features.8'] = output
            layer_to_hook = model_defense.features[8]
            hook = layer_to_hook.register_forward_hook(hook_fn)
        elif args.source_model == 'swin_transformer':
            def hook_fn(module, input, output):
                features['features.3'] = output
            layer_to_hook = model_defense.features[3]
            hook = layer_to_hook.register_forward_hook(hook_fn)

    elif args.defense_method == 'MID':
        if args.source_model == 'mobilenet_v2':
            model_defense_MID = CustomMobileNetV2(pretrained=True)
            local_features_state_dict = model_defense.features.state_dict()
            model_defense_MID.base_model.features.load_state_dict(local_features_state_dict)

            model_defense = model_defense_MID

        elif args.source_model == 'inception_v3':
            
            model_defense_MID = CustomInception3(pretrained=True)
            local_Conv2d_1a_3x3 = model_defense.Conv2d_1a_3x3.state_dict()
            local_Conv2d_2a_3x3 = model_defense.Conv2d_2a_3x3.state_dict()
            local_Conv2d_2b_3x3 = model_defense.Conv2d_2b_3x3.state_dict()
            local_maxpool1 = model_defense.maxpool1.state_dict()
            local_Conv2d_3b_1x1 = model_defense.Conv2d_3b_1x1.state_dict()
            local_Conv2d_4a_3x3 = model_defense.Conv2d_4a_3x3.state_dict()
            local_maxpool2 = model_defense.maxpool2.state_dict()
            local_Mixed_5b = model_defense.Mixed_5b.state_dict()
            local_Mixed_5c = model_defense.Mixed_5c.state_dict()
            local_Mixed_5d = model_defense.Mixed_5d.state_dict()
            local_Mixed_6a = model_defense.Mixed_6a.state_dict()
            local_Mixed_6b = model_defense.Mixed_6b.state_dict()
            local_Mixed_6c = model_defense.Mixed_6c.state_dict()
            local_Mixed_6d = model_defense.Mixed_6d.state_dict()
            local_Mixed_6e = model_defense.Mixed_6e.state_dict()
            local_AuxLogits = model_defense.AuxLogits.state_dict()
            local_Mixed_7a = model_defense.Mixed_7a.state_dict()
            local_Mixed_7b = model_defense.Mixed_7b.state_dict()
            local_Mixed_7c = model_defense.Mixed_7c.state_dict()
            local_dropout = model_defense.dropout.state_dict()
            local_avgpool = model_defense.avgpool.state_dict()

            model_defense_MID.base_model.Conv2d_1a_3x3.load_state_dict(local_Conv2d_1a_3x3)
            model_defense_MID.base_model.Conv2d_2a_3x3.load_state_dict(local_Conv2d_2a_3x3)
            model_defense_MID.base_model.Conv2d_2b_3x3.load_state_dict(local_Conv2d_2b_3x3)
            model_defense_MID.base_model.maxpool1.load_state_dict(local_maxpool1)
            model_defense_MID.base_model.Conv2d_3b_1x1.load_state_dict(local_Conv2d_3b_1x1)
            model_defense_MID.base_model.Conv2d_4a_3x3.load_state_dict(local_Conv2d_4a_3x3)
            model_defense_MID.base_model.maxpool2.load_state_dict(local_maxpool2)
            model_defense_MID.base_model.Mixed_5b.load_state_dict(local_Mixed_5b)
            model_defense_MID.base_model.Mixed_5c.load_state_dict(local_Mixed_5c)
            model_defense_MID.base_model.Mixed_5d.load_state_dict(local_Mixed_5d)
            model_defense_MID.base_model.Mixed_6a.load_state_dict(local_Mixed_6a)
            model_defense_MID.base_model.Mixed_6b.load_state_dict(local_Mixed_6b)
            model_defense_MID.base_model.Mixed_6c.load_state_dict(local_Mixed_6c)
            model_defense_MID.base_model.Mixed_6d.load_state_dict(local_Mixed_6d)
            model_defense_MID.base_model.Mixed_6e.load_state_dict(local_Mixed_6e)
            model_defense_MID.base_model.AuxLogits.load_state_dict(local_AuxLogits)
            model_defense_MID.base_model.Mixed_7a.load_state_dict(local_Mixed_7a)
            model_defense_MID.base_model.Mixed_7b.load_state_dict(local_Mixed_7b)
            model_defense_MID.base_model.Mixed_7c.load_state_dict(local_Mixed_7c)
            model_defense_MID.base_model.dropout.load_state_dict(local_dropout)
            model_defense_MID.base_model.avgpool.load_state_dict(local_avgpool)

            model_defense = model_defense_MID

        elif args.source_model == 'swin_transformer':
            model_defense_MID = CustomSwinTransformer(pretrained=True)
            local_features_state_dict = model_defense.features.state_dict()
            model_defense_MID.base_model.features.load_state_dict(local_features_state_dict)

            model_defense = model_defense_MID

    model_defense = model_defense.to(device)

    optimizer = torch.optim.Adam(model_defense.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        if args.defense_method == 'TL':
            train_loss = train_one_epoch(model_defense, train_loader, optimizer, device, logger)
        elif args.defense_method == 'BiDO':
            train_loss = train_one_epoch_BiDO(args, model_defense, train_loader, optimizer, device, logger)
        elif args.defense_method == 'MID':
            train_loss = train_one_epoch_MID(args, model_defense, train_loader, optimizer, device, logger)
        elif args.defense_method == 'LS':
            train_loss = train_one_epoch_LS(args, model_defense, train_loader, optimizer, device, logger)
    
        test_acc = test(args, model_defense, test_loader, device, logger)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(model_defense.state_dict(), './models/model_defense_best_model.pth')
        #     print("Saved best model")

        save_name = './models/' + args.source_model + '_' + args.defense_method + '_'
        if args.defense_method == 'BiDO':
            save_name = save_name + str(args.coef_hidden_input) + '_' + str(args.coef_hidden_output)
        elif args.defense_method == 'MID':
            save_name = save_name + str(args.beta)
        elif args.defense_method == 'LS':
            save_name = save_name + str(args.coef_label_smoothing)
        elif args.defense_method == 'TL':
            save_name = save_name + str(args.layer_name)

        torch.save(model_defense.state_dict(), save_name)
        print("Saved model.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--defense_method', default=None, type=str)
    parser.add_argument('--source_model', default=None, type=str)

    # BiDO
    parser.add_argument('--coef_hidden_input', type=float, default=0.05)
    parser.add_argument('--coef_hidden_output', type=float, default=0.5)

    # MID
    parser.add_argument('--beta', type=float, default=0.01)

    # LS
    parser.add_argument('--coef_label_smoothing', type=float, default=0.1)

    # TL
    parser.add_argument('--layer_name', type=str, default='features.2')

    args = parser.parse_args()

    LOG_name = './LOG_defense/' + args.defense_method + '/' + args.source_model

    if args.defense_method == 'BiDO':
        args.LOG = LOG_name + '_' + str(args.coef_hidden_input) + '_' + str(args.coef_hidden_output)

    elif args.defense_method == 'MID':
        args.LOG = LOG_name + '_' + str(args.beta)

    elif args.defense_method == 'LS':
        args.LOG = LOG_name + '_' + str(args.coef_label_smoothing)

    elif args.defense_method == 'TL':
        args.LOG = LOG_name + '_' + args.layer_name
    

    main(args)