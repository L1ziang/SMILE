#!/usr/bin/env python3
# coding=utf-8
import torch
from torch import nn
import torchvision.models as models

import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
from facenet_pytorch import InceptionResnetV1
import net_sphere
from typing import Any, Callable, List, Optional, Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from collections import OrderedDict

class CustomMobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomMobileNetV2, self).__init__()
        self.base_model = models.mobilenet.mobilenet_v2(pretrained=True)
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.base_model.last_channel, 1024),
            nn.Linear(512, 8631),
        )

        eps_0 = torch.randn(512).cuda()
        self.eps_0 = eps_0.reshape(1,512)
    
    def forward(self, x):
        x = self.base_model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        statics = self.base_model.classifier[:1](x)
        mu, std = statics[:, : 512], statics[:, 512 : 512 * 2]
        std = F.softplus(std - 5, beta=1)
        feat = mu + std * self.eps_0
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
    

class InceptionAux(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x
    
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


def get_model(arch_name, device, use_dropout=False):
    if arch_name == 'vgg16bn':
        net = vgg_m_face_bn_dag.vgg_m_face_bn_dag('./classification_models/vgg_m_face_bn_dag.pth', use_dropout=use_dropout).to(device)
    elif arch_name == 'resnet50':
        net = resnet50_scratch_dag.resnet50_scratch_dag('./classification_models/resnet50_scratch_dag.pth').to(device)
    elif arch_name == 'vgg16':
        net = vgg_face_dag.vgg_face_dag('./classification_models/vgg_face_dag.pth', use_dropout=use_dropout).to(device)
    elif arch_name == 'inception_resnetv1_vggface2':
        net = InceptionResnetV1(classify=True, pretrained='vggface2').to(device)
    elif arch_name == 'inception_resnetv1_casia':
        net = InceptionResnetV1(classify=True, pretrained='casia-webface').to(device)
    elif arch_name == 'sphere20a':
        net = getattr(net_sphere, 'sphere20a')(use_dropout=use_dropout)
        net.load_state_dict(torch.load('./classification_models/sphere20a_20171020.pth'))
        net.to(device)
    elif arch_name == 'mobilenet_v2':
        net = models.mobilenet.mobilenet_v2(pretrained=False)
        net.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(net.last_channel, 8631),
        )
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/mobilenet_v2_best_model.pth', map_location=device))
        net.to(device)
    elif arch_name == 'efficientnet_b0':
        net = models.efficientnet.efficientnet_b0(pretrained=True)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 8631),
        )
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/efficientnet_b0_best_model.pth', map_location=device))
        net.to(device)
    elif arch_name == 'inception_v3':
        net = models.inception_v3(pretrained=True)
        if hasattr(net, 'AuxLogits'):
            net.AuxLogits = InceptionAux(768, 8631)  
            net.fc = nn.Linear(2048, 8631)
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/inception_v3_best_model.pth', map_location=device))
        net.to(device)
    elif arch_name == 'swin_transformer':
        net = models.swin_transformer.swin_v2_t(pretrained=True)
        net.head = nn.Linear(768, 8631)
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/swin_transformer_best_model.pth', map_location=device))
        net.to(device)
    elif arch_name == 'vgg16':
        net = models.vgg16_bn(pretrained=True)
        net.classifier[6] = nn.Linear(4096, 8631)
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/vgg16_best_model.pth', map_location=device))
        net.to(device)
    elif arch_name == 'vision_transformer':

        net = models.vision_transformer.vit_b_16(pretrained=True)
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(768, 8631)
        net.heads = nn.Sequential(heads_layers)
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/vision_transformer_2_best_model.pth', map_location=device))
        net.to(device)
    elif arch_name == 'efficientnet_b0_casia':
        net = models.efficientnet.efficientnet_b0(pretrained=True)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 10575),
        )
        net.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models_CASIA/efficientnet_b0_best_model.pth', map_location=device))
        net.to(device)

    # To be developed
    elif arch_name == 'cat_resnet18':
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 14)
        net.load_state_dict(torch.load('./cat_breed_data/cat_resnet18.pt'))
        net.to(device)
    elif arch_name == 'car_resnet34':
        net = models.resnet34(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 196)
        net.load_state_dict(torch.load('./car_model_data/new_resnet34.pt'))
        net.to(device)
    else:
        raise AssertionError('wrong arch_name')
    net.eval()
    return net

def get_model_defense(defense_name, arch_name, device, use_dropout=False):
    if arch_name == 'mobilenet_v2':
        if 'MID' in defense_name:
            net = CustomMobileNetV2(pretrained=True)
        else:
            net = models.mobilenet.mobilenet_v2(pretrained=False)
            net.classifier =  nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(net.last_channel, 8631),
            )
        p = '/root/autodl-tmp/train_classification_models/models/' + defense_name
        net.load_state_dict(torch.load(p, map_location=device))
        net.to(device)
    elif arch_name == 'swin_transformer':
        if 'MID' in defense_name:
            net = CustomSwinTransformer(pretrained=True)
        else:
            net = models.swin_transformer.swin_v2_t(pretrained=True)
            net.head = nn.Linear(768, 8631)
        p = '/root/autodl-tmp/train_classification_models/models/' + defense_name
        net.load_state_dict(torch.load(p, map_location=device))
        net.to(device)
    
    net.eval()
    return net

def get_input_resolution(arch_name):
    resolution = 224
    if arch_name.startswith('inception_resnetv1'):
        resolution = 160
    elif arch_name == 'sphere20a':
        resolution = (112, 96)
    elif arch_name.startswith('ccs19ami'):
        resolution = 64
        if 'rgb' not in arch_name:
            pass
    elif arch_name in ['azure', 'clarifai', ]:
        resolution = 256
    elif arch_name == 'car_resnet34':
        resolution = 400

    return resolution
