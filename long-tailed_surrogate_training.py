import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from resnet50_scratch_dag import Resnet50_scratch_dag as resnet50
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
from inceptionresnetv1_4finetune import InceptionResnetV1_4finetune, InceptionResnetV1_4finetune_E
from mobilenetv2_4finetune import MobileNetV2_E
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from my_target_models import get_model, get_input_resolution
from my_utils import crop_and_resize, normalize, ALL_MEANS, ALL_STDS

import torch.nn as nn
import os
import glob
import numpy as np

import net_sphere
from net_sphere import AngleLinear
import resnet50_scratch_dag
import torchvision.models as models
from my_target_models import InceptionAux
from collections import OrderedDict
from torchvision.ops.misc import Conv2dNormActivation
from efficientnetb0_4finetune import EfficientNet_E
from torchvision.models.efficientnet import _efficientnet_conf
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from inceptionv3_4finetune import Inception3_E
from torchvision.models.inception import BasicConv2d, InceptionE
from swintransformer_4finetune import SwinTransformer_E
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2
from torchvision.ops.misc import MLP, Permute
from vitb16_4finetune import VisionTransformer_E

from torchvision.utils import save_image
from tqdm import tqdm

def build_weight_k(all_logits):
    beta = 0.9
    
    topk_indices = torch.topk(all_logits, k=10, dim=1)[1]
    
    flattened_indices = topk_indices.view(-1)
    label_counts = torch.bincount(flattened_indices, minlength=all_logits.shape[1])
    
    weights = torch.zeros_like(label_counts, dtype=torch.float)
    max_count = torch.max(label_counts)
    min_count = torch.min(label_counts)

    for idx, count in enumerate(label_counts):
        weights[idx] = (1-beta)/(1-beta**(count+1)) # (1-beta)/(1-beta**count)
    print('weight: ', weights.shape)
    print(weights)
    print(min(weights),max(weights))
    
    return weights

def count_label_data_ranges(all_logits):
    max_indices = torch.argmax(all_logits, dim=1)
    label_counts = torch.bincount(max_indices)
    topk_indices = torch.topk(all_logits, k=5, dim=1)[1]
    stacked_indices = topk_indices.view(-1)
    stacked_indices = stacked_indices.int()
    index_counts = torch.bincount(stacked_indices, minlength=all_logits.size(1))
    ranges = {
        '0': 0,
        '1-10':0,
        '11-20': 0,
        '21-50': 0,
        '51-100': 0,
        '100+': 0
    }

    for count in label_counts:
        if count == 0:
            ranges['0'] += 1
        elif count<=10:
            ranges['1-10'] +=1
        elif count <= 20:
            ranges['11-20'] += 1
        elif count <= 50:
            ranges['21-50'] += 1
        elif count <= 100:
            ranges['51-100'] += 1
        else:
            ranges['100+'] += 1
        
    return ranges

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        try:
            batch_index = idx // 100
            index_within_batch = idx % 100
            x = self.X[batch_index][index_within_batch]
            x = self.transform(x*255.0) # 乘255.0，使图像tensor的数值范围是[0~255]，这样做之后才能通过transform做归一化
            y = self.Y[idx]
            return x, y
        except Exception as e:
            print(f"Error in loading data at index {idx}: {e}")
        raise

class Normalize(torch.nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image_tensor):
        image_tensor = (image_tensor-torch.tensor(self.mean, device=image_tensor.device)[:, None, None])/torch.tensor(self.std, device=image_tensor.device)[:, None, None]
        return image_tensor

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def diversity_loss(p_i, p_j):
    T = 0.5
    loss = F.kl_div(F.log_softmax(p_i / T, dim=1), F.softmax(p_j / T, dim=1), reduction='batchmean')
    return 1.0/loss

def D_loss(outs):
    p_i = outs[0]
    p_j = outs[1]
    p_k = outs[2]
    diversity_losss = diversity_loss(p_i, p_j)+diversity_loss(p_i, p_k)+diversity_loss(p_j, p_k) / 3.0
    return  diversity_losss

def D_loss_mean(final_out, outs, args):
    loss = 0
    T = 0.5
    for ind in range(args.num_experts):
        stabilized_outs_ind = torch.clamp(outs[ind] / T, -20, 20)
        L = F.kl_div(F.log_softmax(final_out / T, dim=1), F.softmax(stabilized_outs_ind / T, dim=1), reduction='batchmean')
        loss += 1.0 / L
    return  loss

def train(tpk_weights, model, train_loader, optimizer, epoch, device, writer, prefix=100):
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

        data, target = data.to(device), target.to(device)
        weight_tpk = torch.zeros(target.size(0), dtype=torch.float)
        for i in range(target.size(0)):
            max_prob_idx = torch.argmax(target[i])
            weight_tpk[i] = tpk_weights[max_prob_idx]
        weight_tpk = weight_tpk.to(device)

        optimizer.zero_grad()
        if args.arch_name_finetune == 'inception_v3':
            output, x_outs, aux_output, aux_outs = model(data, 1)
            T = 0.5  
            kl_loss = 0
            for ind in range(args.num_experts):
                kl_loss += F.kl_div(F.log_softmax(x_outs[ind] / T, dim=1), F.softmax(target / T, dim=1), reduction='batchmean')
                kl_loss += 0.3 * F.kl_div(F.log_softmax(aux_outs[ind] / T, dim=1), F.softmax(target / T, dim=1), reduction='batchmean')
            _, predicted = torch.max(target, 1)
            criterion = nn.CrossEntropyLoss()
            ce_loss = 0
            for ind in range(args.num_experts):
                ce_loss += criterion(x_outs[ind], predicted)
                ce_loss += criterion(aux_outs[ind], predicted)
        else:
            if args.arch_name_finetune == 'sphere20a':
                exit()
                output = model(data)[0]
            else:
                final_out, outs = model(data)
            T = 0.5
            kl_loss = 0
            for ind in range(args.num_experts):
                kl_loss += F.kl_div(F.log_softmax(outs[ind] / T, dim=1), F.softmax(target / T, dim=1), reduction='batchmean')

            _, predicted = torch.max(target, 1)
            criterion = nn.CrossEntropyLoss()
            ce_loss = 0
            for ind in range(args.num_experts):
                ce_loss += criterion(outs[ind], predicted)
        
        if args.arch_name_finetune == 'inception_v3':
            diversity_loss = args.lambda_diversity * (D_loss_mean(output, x_outs, args) + 0.3 * D_loss_mean(aux_output, aux_outs, args))
        else:
            diversity_loss = args.lambda_diversity * D_loss_mean(final_out, outs, args)

        kl_loss = (kl_loss * weight_tpk).sum()
        ce_loss =( ce_loss * weight_tpk).sum()
        ce_loss = args.lambda_ce * ce_loss
        total_loss = kl_loss + ce_loss + diversity_loss

        total_loss.backward()
        optimizer.step()
        if batch_idx % prefix == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tKL_Loss: {:.6f}  CE_Loss: {:.6f}  Diversity_Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), kl_loss.item(), ce_loss.item(), diversity_loss.item()))
            writer.add_scalar('Loss/train', kl_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss_ce/train', ce_loss.item(), epoch * len(train_loader) + batch_idx)

def test(model, test_loader, device, epoch, writer):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            if args.arch_name_finetune == 'sphere20a':
                output = model(data)[0]
            if args.arch_name_finetune == 'inception_v3':
                output, _, _, _ = model(data, 0)
            else:
                output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            _, predicted_top1 = output.max(1)
            correct_top1 += predicted_top1.eq(target.view_as(predicted_top1)).sum().item()

            _, predicted_top5 = output.topk(5, 1, True, True)
            correct_top5 += predicted_top5.eq(target.view(-1, 1).expand_as(predicted_top5)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy_top1 = 100. * correct_top1 / len(test_loader.dataset)
    accuracy_top5 = 100. * correct_top5 / len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct_top1, len(test_loader.dataset), accuracy_top1, correct_top5, len(test_loader.dataset), accuracy_top5))
    
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/Top-1', accuracy_top1, epoch)
    writer.add_scalar('Accuracy/Top-5', accuracy_top5, epoch)

    return accuracy_top1, accuracy_top5

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    all_logits_file = os.path.join('./blackbox_attack_data',
                                    args.target_dataset,
                                    args.arch_name_target,
                                    args.dataset,
                                    'all_logits.pt')
    all_logits = torch.load(all_logits_file).to(args.device)
    all_logits = all_logits[:args.query_num]
    print('all_logits.shape: ', all_logits.shape)

    if args.dataset == 'celeba_partial256':
        img_dir = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8_25'
    elif args.dataset == 'ffhq':
        img_dir = './stylegan_sample_z_stylegan_ffhq256_0.7_8_25'

    img_files_path = sorted(glob.glob(os.path.join(img_dir, 'sample_*_img.pt')))
    img_files = [torch.load(x).to(device) for x in img_files_path[:int(args.query_num / 100.0)]]
    print('len(img_files): ', len(img_files))
    print('img_files[0].shape: ', img_files[0].shape)

    tpk_weights = build_weight_k(all_logits)

    if args.arch_name_finetune == 'resnet50':
        resolution = get_input_resolution(args.arch_name_finetune)
    elif args.arch_name_finetune == 'inception_resnetv1_vggface2':
        resolution = get_input_resolution(args.arch_name_finetune)
    elif args.arch_name_finetune == 'inception_resnetv1_casia':
        resolution = get_input_resolution(args.arch_name_finetune)
    elif args.arch_name_finetune == 'mobilenet_v2':
        resolution = 224
    elif args.arch_name_finetune == 'efficientnet_b0':
        resolution = 256
    elif args.arch_name_finetune == 'efficientnet_b0_casia':
        resolution = 256
    elif args.arch_name_finetune == 'inception_v3':
        resolution = 342
    elif args.arch_name_finetune == 'swin_transformer':
        resolution = 260
    elif args.arch_name_finetune == 'vision_transformer':
        resolution = 224
    elif args.arch_name_finetune == 'vgg16':
        resolution = 224
    elif args.arch_name_finetune == 'vgg16bn':
        resolution = 224
    elif args.arch_name_finetune == 'sphere20a':
        resolution = get_input_resolution(args.arch_name_finetune)

    Mean = ALL_MEANS[args.arch_name_finetune]
    Std = ALL_STDS[args.arch_name_finetune]

    train_transform = transforms.Compose([
            transforms.Resize(resolution),
            Normalize(Mean, Std)
        ])
    
    dataset = CustomDataset(img_files, all_logits, train_transform)
    traindataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ranges = count_label_data_ranges(all_logits)

    args.resolution = resolution
    if args.target_dataset == "CASIA":
        if args.arch_name_finetune == 'resnet50':
            T_resize = 360
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(args.resolution),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'inception_resnetv1_vggface2':
            T_resize = 210
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(args.resolution),
                Normalize(Mean, Std)
                ])
        elif args.arch_name_finetune == 'mobilenet_v2':
            T_resize = 360
            RESIZE = 224
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'efficientnet_b0':
            T_resize = 360
            RESIZE = 256
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'inception_v3':
            T_resize = 360
            RESIZE = 342
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'swin_transformer':
            T_resize = 360
            RESIZE = 260
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'vgg16':
            T_resize = 360
            RESIZE = 224
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'vision_transformer':
            T_resize = 360
            RESIZE = 224
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(RESIZE),
                Normalize(Mean, Std)
            ])
        
        if args.arch_name_target == 'inception_resnetv1_casia' or args.arch_name_target == 'efficientnet_b0_casia':
            totalset = torchvision.datasets.ImageFolder("/root/autodl-tmp/CASIA-WebFace", transform=test_transform)

        if args.arch_name_target == 'sphere20a':
            test_loader = None
        else:
            trainset_list, testset_list = train_test_split(list(range(len(totalset.samples))), test_size=0.01, random_state=666)
            trainset_no_use= Subset(totalset, trainset_list)
            testset= Subset(totalset, testset_list)
            test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.target_dataset == "vggface":
        if args.arch_name_finetune == 'vgg16': 
            T_resize = 225
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                Normalize(Mean, Std)
            ])
        elif args.arch_name_finetune == 'vgg16bn': 
            T_resize = 225
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                Normalize(Mean, Std)
            ])
        totalset = None
        test_loader = None

    if args.target_dataset == "vggface2":
        if args.arch_name_finetune == 'inception_resnetv1_casia': 
            if args.arch_name_target == 'inception_resnetv1_vggface2':
                T_resize = 220
            elif args.arch_name_target == 'resnet50':
                T_resize = 220
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                Normalize(Mean, Std)
            ])
        if args.arch_name_finetune == 'efficientnet_b0_casia': 
            if args.arch_name_target == 'inception_resnetv1_vggface2':
                T_resize = 300
            elif args.arch_name_target == 'resnet50':
                T_resize = 300
            test_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(T_resize),
                transforms.CenterCrop(args.resolution),
                Normalize(Mean, Std)
            ])

        totalset = torchvision.datasets.ImageFolder("../vggface2/train", transform=test_transform)
        trainset_list, testset_list = train_test_split(list(range(len(totalset.samples))), test_size=0.01, random_state=666)
        trainset_no_use= Subset(totalset, trainset_list)
        testset= Subset(totalset, testset_list)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    left, right = args.finetune_mode.split('->')
    if left == 'vggface':
        num_classification = 2622
    elif left == 'vggface2':
        num_classification = 8631
    elif left == 'CASIA':
        if args.arch_name_target == 'sphere20a':
            num_classification = 10574
        else:
            num_classification = 10575

    if right == 'vggface2':
        if args.arch_name_finetune == 'resnet50':
            model_4finetune = resnet50_scratch_dag.resnet50_scratch_dag_E('./classification_models/resnet50_scratch_dag.pth', args.num_experts)#.to(device)
            model_4finetune.classifier = nn.ModuleList([nn.Conv2d(2048, num_classification, kernel_size=[1, 1], stride=(1, 1)) for _ in range(args.num_experts)])
            model_4finetune.conv5_3_1x1_increase = nn.ModuleList([nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False) for _ in range(args.num_experts)])
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.conv5_3_1x1_increase.parameters():
                param.requires_grad = True
            for param in model_4finetune.classifier.parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.conv5_3_1x1_increase.parameters()},{'params': model_4finetune.classifier.parameters()}], lr=0.001)

        if args.arch_name_finetune == 'inception_resnetv1_vggface2':
            model_4finetune = InceptionResnetV1_4finetune_E(
                classify=True,
                pretrained='vggface2',
                num_classes=num_classification,
                num_experts = args.num_experts
            ).to(device)
            model_4finetune.logits = nn.ModuleList([nn.Linear(512, num_classification) for _ in range(args.num_experts)])
            model_4finetune.last_linear = nn.ModuleList([nn.Linear(1792, 512, bias=False) for _ in range(args.num_experts)])

            model_4finetune = model_4finetune.to(device)
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.last_linear.parameters():
                param.requires_grad = True
            for param in model_4finetune.logits.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([{'params': model_4finetune.last_linear.parameters()}, {'params': model_4finetune.logits.parameters()}], lr=0.001)

        if args.arch_name_finetune == 'mobilenet_v2':
            model_pretrain = models.mobilenet.mobilenet_v2(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model_pretrain.last_channel, 8631),
            )
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/mobilenet_v2_best_model.pth', map_location=device))
            model_4finetune = MobileNetV2_E(num_classes = 8631, num_experts = args.num_experts)
            model_4finetune.load_state_dict(model_pretrain.state_dict())
            model_4finetune.classifier = nn.ModuleList([nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model_4finetune.last_channel, num_classification),
                ) for _ in range(args.num_experts)])
            new_features = list(model_4finetune.features.children())[:-1]
            model_4finetune.features = nn.Sequential(*new_features)
            for _ in range(args.num_experts):
                model_4finetune.features.append(Conv2dNormActivation(
                    320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6
                ))
            
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.classifier.parameters():
                param.requires_grad = True
            for param in model_4finetune.features[-1*args.num_experts:].parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.features[-1*args.num_experts:].parameters()},{'params': model_4finetune.classifier.parameters()}], lr=0.001)

        if args.arch_name_finetune == 'efficientnet_b0':
            model_pretrain = models.efficientnet.efficientnet_b0(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 8631),
            )
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/efficientnet_b0_best_model.pth', map_location=device))
            inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
            model_4finetune = EfficientNet_E(inverted_residual_setting, num_classes = 8631, num_experts = args.num_experts, dropout=0.2)
            model_4finetune.load_state_dict(model_pretrain.state_dict())

            model_4finetune.classifier = nn.ModuleList([nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1280, num_classification),
                ) for _ in range(args.num_experts)])
            new_features = list(model_4finetune.features.children())[:-1]
            model_4finetune.features = nn.Sequential(*new_features)
            for _ in range(args.num_experts):
                model_4finetune.features.append(Conv2dNormActivation(
                    320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU
                ))
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.classifier.parameters():
                param.requires_grad = True
            for param in model_4finetune.features[-1*args.num_experts:].parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.features[-1*args.num_experts:].parameters()},{'params': model_4finetune.classifier.parameters()}], lr=0.001)

        if args.arch_name_finetune == 'inception_v3':
            model_pretrain = models.inception_v3(pretrained=True)
            if hasattr(model_pretrain, 'AuxLogits'):
                model_pretrain.AuxLogits = InceptionAux(768, 8631)  
                model_pretrain.fc = nn.Linear(2048, 8631)
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/inception_v3_best_model.pth', map_location=device))
            model_4finetune = Inception3_E(num_classes=8631, num_experts = args.num_experts)
            model_4finetune.load_state_dict(model_pretrain.state_dict())

            if hasattr(model_4finetune, 'AuxLogits'):
                model_4finetune.AuxLogits = nn.ModuleList([InceptionAux(768, num_classification) for _ in range(args.num_experts)])
                model_4finetune.fc = nn.ModuleList([nn.Linear(2048, num_classification) for _ in range(args.num_experts)])

            model_4finetune.Mixed_7c = nn.ModuleList([InceptionE(2048) for _ in range(args.num_experts)])
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.AuxLogits.parameters():
                param.requires_grad = True
            for param in model_4finetune.fc.parameters():
                param.requires_grad = True
            for param in model_4finetune.Mixed_7c.parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.Mixed_7c.parameters()}, {'params': model_4finetune.AuxLogits.parameters()}, {'params': model_4finetune.fc.parameters()}], lr=0.001)

        if args.arch_name_finetune == 'swin_transformer':
            model_pretrain = models.swin_transformer.swin_v2_t(pretrained=True)
            model_pretrain.head = nn.Linear(768, 8631)
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/swin_transformer_best_model.pth', map_location=device))
            model_4finetune = SwinTransformer_E(num_classes=8631, num_experts = args.num_experts,
                                                patch_size=[4, 4],
                                                embed_dim=96,
                                                depths=[2, 2, 6, 2],
                                                num_heads=[3, 6, 12, 24],
                                                window_size=[8, 8],
                                                stochastic_depth_prob=0.2,
                                                block=SwinTransformerBlockV2,
                                                downsample_layer=PatchMergingV2,)
            model_4finetune.load_state_dict(model_pretrain.state_dict())
            model_4finetune.head = nn.ModuleList([nn.Linear(768, num_classification) for _ in range(args.num_experts)])
            model_4finetune.features[-1][-1].mlp = nn.ModuleList([MLP(768, [3072, 768], activation_layer=nn.GELU, dropout=0.0) for _ in range(args.num_experts)])

            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.head.parameters():
                param.requires_grad = True
            for param in model_4finetune.features[-1][-1].mlp.parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.features[-1][-1].mlp.parameters()},{'params': model_4finetune.head.parameters()}], lr=0.001)

        if args.arch_name_finetune == 'vision_transformer':
            model_pretrain = models.vision_transformer.vit_b_16(pretrained=True)
            heads_layers1: OrderedDict[str, nn.Module] = OrderedDict()
            heads_layers1["head"] = nn.Linear(768, 8631)
            model_pretrain.heads = nn.Sequential(heads_layers1)
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/vision_transformer_2_best_model.pth', map_location=device))
            model_4finetune = VisionTransformer_E(num_classes=8631, num_experts = args.num_experts, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, image_size=224)
            model_4finetune.load_state_dict(model_pretrain.state_dict())
            model_4finetune.heads = nn.ModuleList([nn.Linear(768, num_classification) for _ in range(args.num_experts)])

            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.heads.parameters():
                param.requires_grad = True
            for param in model_4finetune.encoder.layers[-1].mlp.parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.encoder.layers[-1].mlp.parameters()}, {'params': model_4finetune.heads.parameters()}], lr=0.001)
            
    elif right == 'CASIA':
        if args.arch_name_finetune == 'inception_resnetv1_casia':
            model_4finetune = InceptionResnetV1_4finetune_E(
                classify=True,
                pretrained='casia-webface',
                num_classes=num_classification,
                num_experts = args.num_experts
            ).to(device)
            model_4finetune.logits = nn.ModuleList([nn.Linear(512, num_classification) for _ in range(args.num_experts)])
            model_4finetune.last_linear = nn.ModuleList([nn.Linear(1792, 512, bias=False) for _ in range(args.num_experts)])

            model_4finetune = model_4finetune.to(device)
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.last_linear.parameters():
                param.requires_grad = True
            for param in model_4finetune.logits.parameters():
                param.requires_grad = True
            for param in model_4finetune.last_bn.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([{'params': model_4finetune.last_bn.parameters()},{'params': model_4finetune.last_linear.parameters()}, {'params': model_4finetune.logits.parameters()}], lr=0.001)
        if args.arch_name_finetune == 'efficientnet_b0_casia':
            model_pretrain = models.efficientnet.efficientnet_b0(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 10575),
            )
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models_CASIA/efficientnet_b0_best_model_2.pth', map_location=device))
            inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
            model_4finetune = EfficientNet_E(inverted_residual_setting, num_classes = 10575, num_experts = args.num_experts, dropout=0.2)
            model_4finetune.load_state_dict(model_pretrain.state_dict())
            model_4finetune.classifier = nn.ModuleList([nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(1280, num_classification),
                ) for _ in range(args.num_experts)])
            new_features = list(model_4finetune.features.children())[:-1]
            model_4finetune.features = nn.Sequential(*new_features)
            for _ in range(args.num_experts):
                model_4finetune.features.append(Conv2dNormActivation(
                    320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU
                ))
            for param in model_4finetune.parameters():
                param.requires_grad = False
            for param in model_4finetune.classifier.parameters():
                param.requires_grad = True
            for param in model_4finetune.features[-1*args.num_experts:].parameters():
                param.requires_grad = True
            model_4finetune = model_4finetune.to(device)
            optimizer = torch.optim.Adam([{'params': model_4finetune.features[-1*args.num_experts:].parameters()},{'params': model_4finetune.classifier.parameters()}], lr=0.001)
            
    writer = SummaryWriter(f'runs_SMILE/{args.target_dataset}_{args.arch_name_target}_{args.arch_name_finetune}_{args.finetune_mode}_{args.dataset}_{args.query_num}_{args.num_experts}')
    save_path = f'model_checkpoints_SMILE/{args.target_dataset}_{args.arch_name_target}_{args.arch_name_finetune}_{args.finetune_mode}_{args.dataset}_{args.query_num}_{args.num_experts}'
    os.makedirs(save_path, exist_ok=True)

    acc_beat = 0.00001
    epochs = args.epoch
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        train(tpk_weights, model_4finetune, traindataloader, optimizer, epoch, device, writer, prefix=100)

        if test_loader != None:
            if epoch % 25 == 0 or epoch == 199:
                acc, acc_top5 = test(model_4finetune,test_loader,device, epoch, writer)

        torch.save(model_4finetune.state_dict(), f'{save_path}/model_epoch_last.pt')

        if test_loader != None:
            if epoch % 25 == 0 or epoch == 199:
                if acc > acc_beat:
                    acc_beat = acc
                    print("Best ACC.")
                    torch.save(model_4finetune.state_dict(), f'{save_path}/best_model.pt')
    
    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666, help="random seed")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128) # For Resnet50,default=32
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--query_num', type=int, default=2500)
    parser.add_argument('--target_dataset', default='CASIA', type=str, choices=['vggface2', 'vggface', 'CASIA'], help='test dataset')
    parser.add_argument('--dataset', default='celeba_partial256', type=str, choices=['ffhq', 'celeba_partial256'], help='stylegan dataset')
    parser.add_argument('--arch_name_target', default='inception_resnetv1_casia', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name from torchvision or resnet50v15')
    parser.add_argument('--arch_name_finetune', default='inception_resnetv1_vggface2', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name from torchvision or resnet50v15')
    parser.add_argument('--finetune_mode', default='CASIA->vggface2', type=str, choices=['vggface->vggface2', 'vggface->CASIA', 'vggface2->CASIA', 'CASIA->vggface2'], help='mode')

    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--lambda_diversity', type=int, default=10)
    parser.add_argument('--lambda_ce', type=int, default=0.15)

    args = parser.parse_args()
    main(args)