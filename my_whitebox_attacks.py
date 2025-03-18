#!/usr/bin/env python3
# coding=utf-8
import argparse
import glob
import os
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import numpy as np
from torch import optim

import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
from facenet_pytorch import InceptionResnetV1
import net_sphere
from my_utils import normalize, clip_quantile_bound, create_folder, Tee, crop_and_resize, resize_img
from my_target_models import get_model, get_input_resolution

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

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
from resnet50_scratch_dag import Resnet50_scratch_dag as resnet50
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
from inceptionresnetv1_4finetune import InceptionResnetV1_4finetune, InceptionResnetV1_4finetune_E
from mobilenetv2_4finetune import MobileNetV2_E

random.seed(0)

import torchvision.transforms as T
import pickle

def transformations(img, resolution):
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0), ratio=(1.0, 1.0), antialias=True),
    ])
    img = transform(img)
    return img


def adjust_lr(optimizer, initial_lr, epoch, epochs, rampdown=0.25, rampup=0.05):
    # from https://github.com/rosinality/style-based-gan-pytorch/blob/master/projector.py#L45
    t = epoch / epochs
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    lr = initial_lr * lr_ramp

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_input_resolution_self(arch_name):
    resolution = 224
    # to_grayscale = False
    if arch_name.startswith('inception_resnetv1'):
        resolution = 160
    elif arch_name == 'sphere20a':
        resolution = (112, 96)
    elif arch_name.startswith('ccs19ami'):
        resolution = 64
        if 'rgb' not in arch_name:
            # to_grayscale = True
            pass

    elif arch_name == 'mobilenet_v2':
        resolution = 224
    elif arch_name == 'efficientnet_b0':
        resolution = 256
    elif arch_name == 'inception_v3':
        resolution = 342
    elif arch_name == 'swin_transformer':
        resolution = 260
    elif arch_name == 'vision_transformer':
        resolution = 224

    elif arch_name == 'efficientnet_b0_casia':
        resolution = 256
    
    elif arch_name in ['azure', 'clarifai', ]:
        resolution = 256
    elif arch_name == 'car_resnet34':
        resolution = 400

    return resolution

def find_closest_latent(vec, latent_dir):
    min_l2_distance = float('inf')
    closest_img_path = None

    for root, dirs, files in os.walk(latent_dir):
        latent_files = [f for f in files if f.endswith('.latent') and f != 'average.latent']
        
        for latent_file in latent_files:
            latent_path = os.path.join(root, latent_file)

            with open(latent_path, 'rb') as f:
                latent_data = pickle.load(f)
                
            latent_features = latent_data[0]['features']
            
            l2_distance = np.sum((vec - latent_features) ** 2)
            
            if l2_distance < min_l2_distance:
                min_l2_distance = l2_distance
                closest_img_path = latent_data[0]['img_path']
    
    return min_l2_distance, closest_img_path

def compute_conf(args, exp_name, net, arch_name, resolution, targets, imgs):

    try:
        label_logits_dict = torch.load(os.path.join('./centroid_data', arch_name, 'test/centroid_logits.pt'))
    except FileNotFoundError:
        print('Note: centroid_logits.pt is not found')
        label_logits_dict = None

    outputs = net(normalize(crop_and_resize(imgs, arch_name, resolution)*255., arch_name))
    if arch_name == 'sphere20a':
        outputs = outputs[0]
    
    if args.target_dataset != 'vggface':
        if args.test_arch_name == 'sphere20a':
            base_dir = '/root/autodl-tmp/CASIA-WebFace_eval/eval/sphere20a'
        elif args.test_arch_name == 'inception_resnetv1_casia':
            base_dir = '/root/autodl-tmp/CASIA-WebFace_eval/eval/inception_resnetv1_casia'
        elif args.test_arch_name == 'inception_resnetv1_vggface2':
            base_dir = '/root/autodl-tmp/vggface2_eval/eval/inception_resnetv1_vggface2'
        elif args.test_arch_name == 'resnet50':
            base_dir = '/root/autodl-tmp/vggface2_eval/eval/resnet50'
        
        average_latents = []
        for root, dirs, files in os.walk(base_dir):
            if 'average.latent' in files:
                average_latent_path = os.path.join(root, 'average.latent')

                with open(average_latent_path, 'rb') as f:
                    average_latent_data = pickle.load(f)

                average_latents.append(average_latent_data[0]['features'])

        average_latents = average_latents[1:outputs.shape[0]+1] 

        feature_np = features['fs'].detach().cpu().numpy()

        l2_distances = []
        for idx in range(len(feature_np)):
            vec1 = feature_np[idx]
            vec2 = average_latents[idx]
            l2_distance = np.sum((vec1 - vec2) ** 2)

            l2_distances.append(l2_distance)
        l2_distances_mean = np.mean(l2_distances)

        knn_l2_distances = []
        knn_img_paths = []
        label_dirs = sorted([d for d in os.listdir(base_dir)])
        label_dirs = label_dirs[1:outputs.shape[0]+1]

        for idx, feature_vector in enumerate(feature_np):
            label_dir = os.path.join(base_dir, label_dirs[idx])

            min_l2_distance, closest_img_path = find_closest_latent(feature_vector, label_dir)

            knn_l2_distances.append(min_l2_distance)
            knn_img_paths.append(closest_img_path)

        knn_l2_distances_mean = np.mean(knn_l2_distances)    
        
    logits = outputs.cpu()
    logits_softmax = F.softmax(outputs, dim=1)

    target_conf = []
    test_true_list = []

    k = 5
    print(f'top-{k} labels')
    topk_conf, topk_class = torch.topk(outputs, k, dim=1)
    correct_cnt = 0
    topk_correct_cnt = 0
    total_cnt = len(targets)
    l2_dist = []
    for i in range(len(targets)):
        t = targets[i]
        target_conf.append(logits_softmax[i, t].item())
        if label_logits_dict is not None:
            l2_dist.append(torch.dist(logits[i], label_logits_dict[t]).item())
        if topk_class[i][0] == t:
            correct_cnt += 1
            test_true_list.append(i)
        if t in topk_class[i]:
            topk_correct_cnt += 1
    print('target conf:', target_conf)
    l2_dist = l2_dist and sum(l2_dist)/len(l2_dist)
    print(arch_name)
    print(f'top1 acc: {correct_cnt}/{total_cnt} = {correct_cnt/total_cnt:.4f}')
    print(f'topk acc: {topk_correct_cnt}/{total_cnt} = {topk_correct_cnt/total_cnt:.4f}')
    
    formatted_exp_name = exp_name.replace('./', '')

    path = './tmp/' + formatted_exp_name + '.txt'
    os.makedirs(os.path.dirname(path[:-1]), exist_ok=True)

    with open(path, 'w') as f:
        f.write(f"Experiment name: {exp_name}\n")
        f.write(f'top1 acc: {correct_cnt}/{total_cnt} = {correct_cnt/total_cnt:.4f}\n')
        f.write(f'topk acc: {topk_correct_cnt}/{total_cnt} = {topk_correct_cnt/total_cnt:.4f}\n')
        if args.target_dataset != 'vggface':
            f.write(f"l2 distances mean: {l2_distances_mean}\n")
            f.write(f"knn distances mean: {knn_l2_distances_mean}\n")
            f.write(f"image path:\n")
        f.write(str(test_true_list))
        if args.target_dataset != 'vggface':
            f.write(f"{knn_img_paths}\n")

    return test_true_list, target_conf

def add_conf_to_tensors_custom(tensors, confs, testornot, test_true_list=None, color=torch.tensor([1., 0., 0.]).unsqueeze(1).unsqueeze(1), highlight_conf=None):
    """ Note: will clone the tensors to cpu
    """
    print(test_true_list)
    if len(tensors) != len(confs):
        raise AssertionError(f'{len(tensors)} != {len(confs)}, tensors.shape: {tensors.shape}')
    tensors = tensors.detach().cpu().clone()
    if highlight_conf is not None:
        highlight_confs = [x>=highlight_conf for x in confs]
    else:
        highlight_confs = [False] * len(confs)
    if test_true_list is not None:
        highlight_confs = [False] * len(confs)
        for i in range(len(confs)):
            if i in test_true_list:
                highlight_confs[i] = True

    confs = [f'{x:.4f}' for x in confs]
    if testornot == True:
        color=torch.tensor([0., 1., 0.]).unsqueeze(1).unsqueeze(1)
    else:
        color=torch.tensor([1., 0., 0.]).unsqueeze(1).unsqueeze(1)
    tensors_og = tensors.clone()
    for i in range(len(tensors)):
        add_conf_to_tensor_custom(tensors[i], confs[i], color, highlight_confs[i], testornot)
    for i in range(len(tensors_og)):
        add_conf_to_tensor_custom(tensors_og[i], confs[i], color, False, testornot)
    return tensors, tensors_og

def add_conf_to_tensor_custom(tensor, conf, color, highlight, testornot):
    """ Note: in-place modification on tensor
    """
    CONF_MASKS = torch.load('./conf_mask.pt')
    assert tensor.ndim == 3 and tensor.shape[0] == 3, 'tensor shape should be 3xHxW'
    mask = CONF_MASKS[conf]
    if testornot == True:
        tensor[:, -46:-10, 140:250] = (1.-mask) * tensor[:, -46:-10, 140:250] + mask * color
    else:
        tensor[:, -46:-10, 10:120] = (1.-mask) * tensor[:, -46:-10, 10:120] + mask * color

    if highlight:
        width = 5
        tensor[0, :width, :] = 1.
        tensor[0, -width:, :] = 1.
        tensor[0, :, :width] = 1.
        tensor[0, :, -width:] = 1.

class Sample:
    def __init__(self, value, fitness_score=-1):
        """
        value is a tensor
        """
        self.value = value
        self.fitness_score = fitness_score

class VectorizedPopulation_w:
    def __init__(self, population, fitness_scores, apply_noise_func, clip_func, compute_fitness_func, clip_array_func):
        """
        population is a tensor with size N,512
        fitness_scores is a tensor with size N
        """
        self.population = population
        self.fitness_scores = fitness_scores
        self.apply_noise_func = apply_noise_func
        self.clip_func = clip_func
        self.compute_fitness_func = compute_fitness_func
        self.clip_array_func = clip_array_func

    def find_elite(self, index=0):
        self.fitness_scores, indices = torch.sort(self.fitness_scores, dim=0, descending=True)
        self.population = self.population[indices]
        return Sample(self.population[index].detach().clone(), self.fitness_scores[index].item())

    def visualize_imgs(self, filename, generate_images_func, k=8):
        ws = self.population[:k]
        out = generate_images_func(ws, raw_img=True)
        vutils.save_image(out, filename)

def init_population_mirror_w(args):
    """
    find args.n images with highest confidence
    """
    if args.dataset == 'celeba_partial256':
        all_ws_pt_file = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8_25/stylegan_celeba_partial256_all_ws.pt'
    elif args.dataset == 'ffhq':
        all_ws_pt_file = './stylegan_sample_z_stylegan_ffhq256_0.7_8_25/stylegan_ffhq256_all_ws.pt'

    # compute bound in p space
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)
    lrelu = nn.LeakyReLU(negative_slope=0.2)

    all_ws = torch.load(all_ws_pt_file)
    all_ws = all_ws[:args.population_size].to(args.device)
    
    print(f'all_ws.shape: {all_ws.shape}')
    all_ps = invert_lrelu(all_ws)
    all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
    all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
    all_p_mins = all_p_means - args.p_std_ce * all_p_stds
    all_p_maxs = all_p_means + args.p_std_ce * all_p_stds
    all_w_mins = lrelu(all_p_mins)
    all_w_maxs = lrelu(all_p_maxs)
    print(f'all_w_mins.shape: {all_w_mins.shape}')

    def clip_func(w):
        assert w.ndim == 2
        return clip_quantile_bound(w, all_w_mins, all_w_maxs)
    
    def clip_array_func(inputs):
        clipped = np.clip(inputs, all_w_mins.cpu().numpy()[0], all_w_maxs.cpu().numpy()[0])
        return clipped
    
    def apply_noise_func(w, mask, ce):
        assert w.ndim == 2
        p = invert_lrelu(w)
        noise = (2*all_p_stds) * torch.rand_like(all_p_stds) - all_p_stds
        noise = ce * noise
        p = p + mask*noise
        w = lrelu(p)
        return w

    all_logits_file = os.path.join('./blackbox_attack_data',
                                    args.target_dataset,
                                    args.arch_name_target,
                                    args.dataset,
                                    'all_logits.pt')
    all_logits = torch.load(all_logits_file)
    all_logits = all_logits[:args.population_size].to(args.device)
    all_prediction = F.log_softmax(all_logits, dim=1)[:, args.target]

    print('all_prediction.shape', all_prediction.shape, 'device', all_prediction.device)
    topk_conf, topk_ind = torch.topk(all_prediction, args.population_size, dim=0, largest=True, sorted=True)
    population = all_ws[topk_ind].detach().clone()
    fitness_scores = topk_conf
    return VectorizedPopulation_w(population, fitness_scores, apply_noise_func, clip_func, args.compute_fitness_func, clip_array_func)

def my_attack_mirror_w(args, generator, generate_images_func):

    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)

    writer = SummaryWriter(folder_path)
    Tee(os.path.join(folder_path, 'output.log'), 'w')

    test_model = get_model(args.test_arch_name, args.device)
    test_model = test_model.to(args.device)

    population = init_population_mirror_w(args)
    elite = population.find_elite(args.index)
    L = elite.value.unsqueeze(0).clone().to(args.device)

    L.requires_grad = True
    optimizer = optim.Adam([L, ], lr=args.lr, betas=[0.9, 0.999], eps=1e-8)

    criterion = nn.CrossEntropyLoss()
    TARGET = torch.LongTensor([args.target]).to(args.device)

    pbar = tqdm(range(args.epochs+1))
    for epoch in pbar:
        img = crop_and_resize(args.generator(L), args.arch_name_target, args.resolution)

        optimizer.zero_grad()
        args.generator.zero_grad()

        assert img.ndim == 4
        outputs = args.net(normalize(img*255., args.arch_name_target))

        if args.arch_name_target == 'sphere20a':
            outputs = outputs[0]
        
        loss = criterion(outputs, TARGET)

        loss.backward()
        optimizer.step()

        L.data = population.clip_func(L.data)

        with torch.no_grad():
            img1 = generate_images_func(L, raw_img=True)

            outputs = test_model(normalize(crop_and_resize(img1, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))

            if args.test_arch_name == 'sphere20a':
                outputs = outputs[0]

            logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]

            rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

            if epoch % 10 == 0 or epoch == args.epochs-1:
                writer.add_scalar('Target Score', loss.item(), epoch+1)
                writer.add_scalar('Evaluation Score', logits_softmax.item(), epoch+1)
                writer.add_scalar('Rank of label', rank_of_label.item(), epoch+1)

                writer.add_image('Generated Image', img1.squeeze(), global_step=epoch)

    res = Sample(L.detach().clone(), logits_softmax.item())
    torch.save(res, os.path.join(folder_path, 'final_w.pt'))

    return

def init_population_ours_w(args):
    """
    find args.n images with highest confidence
    """
    if args.dataset == 'celeba_partial256':
        all_ws_pt_file = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8_25/stylegan_celeba_partial256_all_ws.pt'
    elif args.dataset == 'ffhq':
        all_ws_pt_file = './stylegan_sample_z_stylegan_ffhq256_0.7_8_25/stylegan_ffhq256_all_ws.pt'

    # compute bound in p space
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)
    lrelu = nn.LeakyReLU(negative_slope=0.2)

    all_ws = torch.load(all_ws_pt_file)
    all_ws = all_ws[:args.population_size].to(args.device)
    
    print(f'all_ws.shape: {all_ws.shape}')
    all_ps = invert_lrelu(all_ws)
    all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
    all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
    all_p_mins = all_p_means - args.p_std_ce * all_p_stds #*1.4
    all_p_maxs = all_p_means + args.p_std_ce * all_p_stds #*1.4
    all_w_mins = lrelu(all_p_mins)
    all_w_maxs = lrelu(all_p_maxs)
    print(f'all_w_mins.shape: {all_w_mins.shape}')

    def clip_func(w):
        assert w.ndim == 2
        return clip_quantile_bound(w, all_w_mins, all_w_maxs)
    
    def clip_array_func(inputs):
        clipped = np.clip(inputs, all_w_mins.cpu().numpy()[0], all_w_maxs.cpu().numpy()[0])
        return clipped
    
    def apply_noise_func(w, mask, ce):
        assert w.ndim == 2
        p = invert_lrelu(w)
        noise = (2*all_p_stds) * torch.rand_like(all_p_stds) - all_p_stds
        noise = ce * noise
        p = p + mask*noise
        w = lrelu(p)
        return w

    all_logits_file = os.path.join('./blackbox_attack_data',
                                    args.target_dataset,
                                    args.arch_name_target,
                                    args.dataset,
                                    'all_logits.pt')
    all_logits = torch.load(all_logits_file)
    all_logits = all_logits[:args.population_size].to(args.device)
    all_prediction = F.log_softmax(all_logits, dim=1)[:, args.target]

    print('all_prediction.shape', all_prediction.shape, 'device', all_prediction.device)
    topk_conf, topk_ind = torch.topk(all_prediction, args.population_size, dim=0, largest=True, sorted=True)
    population = all_ws[topk_ind].detach().clone()
    fitness_scores = topk_conf
    return VectorizedPopulation_w(population, fitness_scores, apply_noise_func, clip_func, args.compute_fitness_func, clip_array_func)


def my_attack_ours_w(args, generator, generate_images_func):

    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)

    writer = SummaryWriter(folder_path)
    Tee(os.path.join(folder_path, 'output.log'), 'w')

    test_model = get_model(args.test_arch_name, args.device)
    test_model = test_model.to(args.device)

    population = init_population_ours_w(args)

    elite = population.find_elite(args.index)
    L = elite.value.unsqueeze(0).clone().to(args.device)

    L.requires_grad = True
    optimizer = optim.Adam([L, ], lr=args.lr, betas=[0.9, 0.999], eps=1e-8)

    criterion = nn.CrossEntropyLoss()
    TARGET = torch.LongTensor([args.target]).to(args.device)

    pbar = tqdm(range(args.epochs+1))
    for epoch in pbar:
        _lr = adjust_lr(optimizer, args.lr, epoch, args.epochs)
        img = crop_and_resize(args.generator(L), args.arch_name_finetune, args.resolution)

        optimizer.zero_grad()
        args.generator.zero_grad()

        assert img.ndim == 4

        if args.arch_name_finetune == 'inception_v3':
            outputs, _, _, _ = args.net(normalize(img*255., args.arch_name_finetune), 0)
        else:
            outputs, _ = args.net(normalize(img*255., args.arch_name_finetune))

        loss = criterion(outputs, TARGET)

        loss.backward()
        optimizer.step()

        L.data = population.clip_func(L.data)

        with torch.no_grad():
            img1 = generate_images_func(L, raw_img=True)

            outputs = test_model(normalize(crop_and_resize(img1, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))

            if args.test_arch_name == 'sphere20a':
                outputs = outputs[0]

            logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]

            rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

            if epoch % 1 == 0 or epoch == args.epochs-1:
                writer.add_scalar('Target Score', loss.item(), epoch+1)
                writer.add_scalar('Evaluation Score', logits_softmax.item(), epoch+1)
                writer.add_scalar('Rank of label', rank_of_label.item(), epoch+1)

                writer.add_image('Generated Image', img1.squeeze(), global_step=epoch)

    res = Sample(L.detach().clone(), logits_softmax.item())
    torch.save(res, os.path.join(folder_path, 'final_w.pt'))

    return

def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(
        torch.eye(outputs.shape[-1], device=outputs.device)[targets] - xi, 0,
        1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1)**2
    v_norm_squared = torch.norm(v, p=2, dim=1)**2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1)**2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss

def my_attack_PPA(args, generator, generate_images_func):
    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)

    writer = SummaryWriter(folder_path)
    Tee(os.path.join(folder_path, 'output.log'), 'w')

    test_model = get_model(args.test_arch_name, args.device)
    test_model = test_model.to(args.device)

    population = init_population_mirror_w(args)

    candidates = []

    for index in range(args.candidate):
        elite = population.find_elite(index)
        L = elite.value.unsqueeze(0).clone().to(args.device)

        L.requires_grad = True
        optimizer = optim.Adam([L, ], lr=args.lr, betas=[0.1, 0.1])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

        TARGET = torch.LongTensor([args.target]).to(args.device)
        pbar = tqdm(range(args.epochs+1))
        for epoch in pbar:
            _lr = adjust_lr(optimizer, args.lr, epoch, args.epochs)
            img = transformations(args.generator(L), args.resolution)

            optimizer.zero_grad()
            args.generator.zero_grad()

            assert img.ndim == 4
            outputs = args.net(normalize(img*255., args.arch_name_target))

            if args.arch_name_target == 'sphere20a':
                outputs = outputs[0]

            loss = poincare_loss(outputs, TARGET).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if index == 0:
                with torch.no_grad():
                    img1 = generate_images_func(L, raw_img=True)

                    outputs = test_model(normalize(crop_and_resize(img1, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))

                    if args.test_arch_name == 'sphere20a':
                        outputs = outputs[0]

                    logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]

                    rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

                    if epoch % 10 == 0 or epoch == args.epochs-1:
                        writer.add_scalar('Target Score', loss.item(), epoch+1)
                        writer.add_scalar('Evaluation Score', logits_softmax.item(), epoch+1)
                        writer.add_scalar('Rank of label', rank_of_label.item(), epoch+1)

                        writer.add_image('Generated Image', img1.squeeze(), global_step=epoch)

        candidates.append(L)

    if args.arch_name_target == 'sphere20a':
        transforms = T.Compose([
            T.RandomResizedCrop(size=args.resolution,
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),
            T.RandomHorizontalFlip(0.5)
        ])
    else:
        transforms = T.Compose([
            T.RandomResizedCrop(size=(args.resolution, args.resolution),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),
            T.RandomHorizontalFlip(0.5)
        ])

    with torch.no_grad():
        mean_score_beat = float('-inf')
        best_i = -1

        for c in range(len(candidates)):
            score = 0
            L_now = candidates[c].detach()
            img2 = args.generator(L_now)

            for i in range(args.iterations):
                img = transforms(img2)
                outputs = args.net(normalize(img*255., args.arch_name_target))
                if args.arch_name_target == 'sphere20a':
                    outputs = outputs[0]

                logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]
                score += logits_softmax
            
            mean_score = score.mean()

            if mean_score > mean_score_beat:
                mean_score_beat = mean_score
                best_i = c

    res = Sample(candidates[best_i].detach().clone(), mean_score_beat)
    torch.save(res, os.path.join(folder_path, 'final_w.pt'))

    return

def load_finetune_model(args):
    if args.target_dataset == 'vggface2':
        save_path = f'vggface2_target/model_checkpoints_{args.EorOG}/{args.target_dataset}_{args.arch_name_target}_{args.arch_name_finetune}_{args.finetune_mode}_{args.dataset}_{args.population_size}_{args.num_experts}'
    else:
        save_path = f'model_checkpoints_{args.EorOG}/{args.target_dataset}_{args.arch_name_target}_{args.arch_name_finetune}_{args.finetune_mode}_{args.dataset}_{args.population_size}_{args.num_experts}'
    
    print("finetune_model loaded.")
    print("Path: ", save_path)

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
            
        if args.arch_name_finetune == 'inception_resnetv1_vggface2':
            model_4finetune = InceptionResnetV1_4finetune_E(
                classify=True,
                pretrained='vggface2',
                num_classes=num_classification,
                num_experts = args.num_experts
            ).to(args.device)
            model_4finetune.logits = nn.ModuleList([nn.Linear(512, num_classification) for _ in range(args.num_experts)])
            model_4finetune.last_linear = nn.ModuleList([nn.Linear(1792, 512, bias=False) for _ in range(args.num_experts)])

        if args.arch_name_finetune == 'mobilenet_v2':
            model_pretrain = models.mobilenet.mobilenet_v2(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model_pretrain.last_channel, 8631),
            )
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/mobilenet_v2_best_model.pth', map_location=args.device))
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
                
        if args.arch_name_finetune == 'efficientnet_b0':
            model_pretrain = models.efficientnet.efficientnet_b0(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 8631),
            )
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/efficientnet_b0_best_model.pth', map_location=args.device))
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

        if args.arch_name_finetune == 'inception_v3':
            model_pretrain = models.inception_v3(pretrained=True)
            if hasattr(model_pretrain, 'AuxLogits'):
                model_pretrain.AuxLogits = InceptionAux(768, 8631)  
                model_pretrain.fc = nn.Linear(2048, 8631)
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/inception_v3_best_model.pth', map_location=args.device))
            model_4finetune = Inception3_E(num_classes=8631, num_experts = args.num_experts)
            model_4finetune.load_state_dict(model_pretrain.state_dict())

            if hasattr(model_4finetune, 'AuxLogits'):
                model_4finetune.AuxLogits = nn.ModuleList([InceptionAux(768, num_classification) for _ in range(args.num_experts)])
                model_4finetune.fc = nn.ModuleList([nn.Linear(2048, num_classification) for _ in range(args.num_experts)])

            model_4finetune.Mixed_7c = nn.ModuleList([InceptionE(2048) for _ in range(args.num_experts)])

        if args.arch_name_finetune == 'swin_transformer':
            model_pretrain = models.swin_transformer.swin_v2_t(pretrained=True)
            model_pretrain.head = nn.Linear(768, 8631)
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/swin_transformer_best_model.pth', map_location=args.device))
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

        if args.arch_name_finetune == 'vision_transformer':
            model_pretrain = models.vision_transformer.vit_b_16(pretrained=True)
            heads_layers1: OrderedDict[str, nn.Module] = OrderedDict()
            heads_layers1["head"] = nn.Linear(768, 8631)
            model_pretrain.heads = nn.Sequential(heads_layers1)
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models/vision_transformer_2_best_model.pth', map_location=args.device))
            model_4finetune = VisionTransformer_E(num_classes=8631, num_experts = args.num_experts, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, image_size=224)
            model_4finetune.load_state_dict(model_pretrain.state_dict())
            model_4finetune.heads = nn.ModuleList([nn.Linear(768, num_classification) for _ in range(args.num_experts)])

    elif right == 'CASIA':
        if args.arch_name_finetune == 'inception_resnetv1_casia':
            model_4finetune = InceptionResnetV1_4finetune_E(
                classify=True,
                pretrained='casia-webface',
                num_classes=num_classification,
                num_experts = args.num_experts
            ).to(args.device)
            model_4finetune.logits = nn.ModuleList([nn.Linear(512, num_classification) for _ in range(args.num_experts)])
            model_4finetune.last_linear = nn.ModuleList([nn.Linear(1792, 512, bias=False) for _ in range(args.num_experts)])

        if args.arch_name_finetune == 'efficientnet_b0_casia':
            model_pretrain = models.efficientnet.efficientnet_b0(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 10575),
            )
            model_pretrain.load_state_dict(torch.load('/root/autodl-tmp/train_classification_models/models_CASIA/efficientnet_b0_best_model_2.pth', map_location=args.device))
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

    if args.arch_name_finetune != 'resnet50':
        model_4finetune.load_state_dict(torch.load(save_path+'/model_epoch_last.pt', map_location=args.device))
    elif args.arch_name_finetune == 'resnet50':
        if args.arch_name_target != 'sphere20a':
            model_4finetune.load_state_dict(torch.load(save_path+'/best_model.pt', map_location=args.device))
        else:
            model_4finetune.load_state_dict(torch.load(save_path+'/model_epoch_last.pt', map_location=args.device))
    return model_4finetune

features = {}

def main(args):

    torch.backends.cudnn.benchmark = True

    print(args)
    print(datetime.now())

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f'using device: {device}')

    if args.attack_mode != 'ours-w':
        net = get_model(args.arch_name_target, device)
        net = net.to(device)
        net.eval()

        args.net = net

    # VGGFace
    if args.arch_name_target == 'vgg16bn':
        args.test_arch_name = 'vgg16'
        result_dir = 'vggface_vgg16bn'
    elif args.arch_name_target == 'vgg16':
        args.test_arch_name = 'vgg16bn'
        result_dir = 'vggface_vgg16'

    # VGGFace2
    elif args.arch_name_target == 'resnet50':
        args.test_arch_name = 'inception_resnetv1_vggface2'
        result_dir = 'vggface2_resnet50'
    elif args.arch_name_target == 'inception_resnetv1_vggface2':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_inceptionrnv1'
    
    elif args.arch_name_target == 'mobilenet_v2':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_mobilenet_v2'
    elif args.arch_name_target == 'efficientnet_b0':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_efficientnet_b0'
    elif args.arch_name_target == 'inception_v3':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_inception_v3'
    elif args.arch_name_target == 'swin_transformer':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_swin_transformer'
    elif args.arch_name_target == 'vision_transformer':
        args.test_arch_name = 'resnet50'
        result_dir = 'vggface2_vision_transformer'

    elif args.arch_name_target == 'inception_resnetv1_casia':
        args.test_arch_name = 'sphere20a'
        result_dir = 'casia_inceptionrnv1'
    elif args.arch_name_target == 'efficientnet_b0_casia':
        args.test_arch_name = 'inception_resnetv1_casia'
        result_dir = 'casia_efficientnet_b0'

    elif args.arch_name_target == 'sphere20a':
        args.test_arch_name = 'inception_resnetv1_casia'

    else:
        raise AssertionError('wrong arch_name')
    
    args.resolution = get_input_resolution_self(args.arch_name_target)
    args.test_resolution = get_input_resolution_self(args.test_arch_name)

    if args.attack_mode == 'Mirror-w'\
        or args.attack_mode == 'PPA' \
        or args.attack_mode == 'ours-w':
        use_w_space = True
        repeat_w = True  # if False, opt w+ space
        # num_layers = 14  # 14 for stylegan w+ space
        # num_layers = 18  # 14 for stylegan w+ space with stylegan_celebahq1024

        if args.dataset == 'celeba_partial256':
            genforce_model = 'stylegan_celeba_partial256'
        elif args.dataset == 'ffhq':
            genforce_model = 'stylegan_ffhq256'
        if not genforce_model.startswith('stylegan'):
            use_w_space = False

        def get_generator(batch_size, device):
            from genforce import my_get_GD
            use_discri = False
            generator, discri = my_get_GD.main(device, genforce_model, batch_size, batch_size, use_w_space=use_w_space, use_discri=use_discri, repeat_w=repeat_w)
            return generator

        generator = get_generator(args.bs, device)
        args.generator = generator

        def generate_images_func(w, raw_img=False):
            assert w.ndim == 2
            if raw_img:
                return generator(w.to(device))
            img = crop_and_resize(generator(w.to(device)), args.arch_name_target, args.resolution)
            return img

        def compute_fitness_func(w):
            img = generate_images_func(w)
            assert img.ndim == 4
            pred = F.log_softmax(args.net(normalize(img*255., args.arch_name_target)), dim=1)

            score = pred[:, args.target]
            return score 
        args.compute_fitness_func = compute_fitness_func

        path_exp = './white_attack/' + args.attack_mode
        if args.attack_mode == 'Mirror-w':
            exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' +  str(args.population_size) + '-' + str(args.epochs) + '-' + str(args.lr)
        elif args.attack_mode == 'PPA':
            exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' +  str(args.population_size) + '-' + str(args.epochs) + '-' + str(args.lr) + '-' + str(args.candidate) + '-' + str(args.iterations)
        elif args.attack_mode == 'ours-w':
            exp_name_tmp = args.EorOG + '-' + args.arch_name_target + '-' + args.dataset + '-' +  str(args.population_size) + '-' + str(args.epochs) + '-' + str(args.lr)+ '-' + str(args.arch_name_finetune) + '-' + str(args.finetune_mode) + '-' + str(args.num_experts)

        
        args.exp_name = os.path.join(path_exp, exp_name_tmp)
        create_folder(args.exp_name)

        if args.test_only:
            if args.attack_mode == 'Mirror-w' \
            or args.attack_mode == 'PPA' \
            or args.attack_mode == 'ours-w':
                test_net = get_model(args.test_arch_name, device)
                # VGGFace
                if args.test_arch_name == 'vgg16':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.relu7
                    hook = layer_to_hook.register_forward_hook(hook_fn)
                
                elif args.test_arch_name == 'vgg16bn':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.relu7
                    hook = layer_to_hook.register_forward_hook(hook_fn)

                # VGGFace2
                elif args.test_arch_name == 'inception_resnetv1_vggface2':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.last_bn
                    hook = layer_to_hook.register_forward_hook(hook_fn)
                
                elif args.test_arch_name == 'resnet50':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.pool5_7x7_s1
                    hook = layer_to_hook.register_forward_hook(hook_fn)

                elif args.test_arch_name == 'sphere20a':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.fc5
                    hook = layer_to_hook.register_forward_hook(hook_fn)
                    
                elif args.test_arch_name == 'inception_resnetv1_casia':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.last_bn
                    hook = layer_to_hook.register_forward_hook(hook_fn)
                
                targets = list(map(int, args.test_target.split(',')))
                ws = []
                all_confs = []
                ws_og = []
                for t in targets:
                    folder_path = args.exp_name + '/LOGS/' + str(t) +'/'+ str(args.index)
                    final_sample = torch.load(os.path.join(folder_path, 'final_w.pt'))
                    w = final_sample.value
                    ws.append(w.squeeze().to(device))
                    score = math.exp(final_sample.fitness_score)
                    all_confs.append(score)
                ws = torch.stack(ws, dim=0)
                imgs = generate_images_func(ws, raw_img=True)
                test_true_list, target_conf_test = compute_conf(args, args.exp_name, test_net, args.test_arch_name, args.test_resolution, targets, imgs)

                imgs, imgs_origanal = add_conf_to_tensors_custom(imgs, target_conf_test, test_true_list=test_true_list, testornot=True)
                create_folder('./tmp')
                formatted_exp_name = args.exp_name.replace('./', '')
                path_png = './tmp/' + formatted_exp_name + '.png'
                path_png_og = './tmp/' + formatted_exp_name + '_og.png'
                vutils.save_image(imgs, path_png, nrow=1)
                vutils.save_image(imgs_origanal, path_png_og, nrow=1)
            
                return
    
        if args.attack_mode == 'Mirror-w':
            args.index = 0
            my_attack_mirror_w(args, generator, generate_images_func)
        elif args.attack_mode == 'PPA':
            args.index = 0
            my_attack_PPA(args, generator, generate_images_func)
        elif args.attack_mode == 'ours-w':
            args.index = 0
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

            args.resolution = resolution

            if args.EorOG != 'OG':
                args.net = load_finetune_model(args)
            else:
                pass
            args.net.eval()
            args.net = args.net.to(args.device)
            my_attack_ours_w(args, generator, generate_images_func)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_mode', type=str, choices=['Mirror-w', 'PPA', 'ours-w'], help='white-box attack mode')
    parser.add_argument('--seed', type=int, default=666, help="random seed")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default=None, help='name')

    parser.add_argument('--target_dataset', default='CASIA', type=str, choices=['vggface2', 'vggface', 'CASIA'], help='target dataset')
    parser.add_argument('--dataset', default='celeba_partial256', type=str, choices=['ffhq', 'celeba_partial256', 'celeba_RLBMI', 'ffhq_RLBMI'], help='stylegan dataset')
    parser.add_argument('--arch_name_target', default='inception_resnetv1_casia', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name')

    parser.add_argument('--bs', default=8, type=int, help='batch size')

    parser.add_argument('--target', default=8, type=int, help='the target label')

    # parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std')

    parser.add_argument('--index', default=0, type=int, help='Top-k index of initialization')

    # Mirror-w, PPA together
    # Mirror-w default
    # parser.add_argument('--population_size', default=20000, type=int, help='population size')
    # parser.add_argument('--epochs', default=1000, type=int, help='optimization epochs')
    # parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')

    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    
    # PPA default
    # parser.add_argument('--population_size', default=20000, type=int, help='population size')
    # parser.add_argument('--lr', type=float, default=0.005, help='learning rate for optimization')
    parser.add_argument('--epochs', default=50, type=int, help='optimization epochs')

    # for Mirror-w
    # parser.add_argument('--population_size', default=20000, type=int, help='population size')
    # parser.add_argument('--epochs', default=1000, type=int, help='optimization epochs')
    # parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    # parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--loss_class_ce', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std; set 0. to unbound')
    # parser.add_argument('--trunc_psi', type=float, default=0.7, help='truncation percentage')
    # parser.add_argument('--trunc_layers', type=int, default=8, help='num of layers to truncate')
    # parser.add_argument('--latent_space', default='w', choices=['w', 'z', 'w+', 'z+'], help='evaluate batch with another model')
    # parser.add_argument('--use_w_mean', action='store_true', help='start optimizing with w_mean')
    # parser.add_argument('--to_truncate_z', action='store_true', help='truncate z vectors')
    # parser.add_argument('--z_std_ce', type=float, default=1., help='set the bound for z space bound mean+-x*std')
    # parser.add_argument('--loss_discri_ce', type=float, default=0., help='coefficient for discri loss')
    # parser.add_argument('--naive_clip_w_bound', default=0., type=float, help='use naive clip in w')
    # parser.add_argument('--energy', action='store_true', help='use energy term')
    # parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    # parser.add_argument('--loss_l2_bound_latent_ce', type=float, default=0., help='ce to bound l2 distance between the optimized latent vectors and the starting vectors.')
    # parser.add_argument('--save_every', type=int, default=100, help='how often to save the intermediate results')
    

    # ours-w
    parser.add_argument('--EorOG', default='SMILE', type=str, choices=['E', 'E_ce', 'SMILE', 'OG'], help='mode')
    # parser.add_argument('--loss_class_ce', type=float, default=1.0, help='coefficient for the main loss in optimization')
    # parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std; set 0. to unbound')
    parser.add_argument('--arch_name_finetune', default='inception_resnetv1_vggface2', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name from torchvision or resnet50v15')
    parser.add_argument('--finetune_mode', default='CASIA->vggface2', type=str, choices=['vggface->vggface2', 'vggface->CASIA', 'vggface2->CASIA', 'CASIA->vggface2'], help='mode')
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--population_size', default=10000, type=int, help='population size')
    # parser.add_argument('--lambda_diversity', type=int, default=10)
    # parser.add_argument('--lambda_ce', type=int, default=0.15)

    # for PPA
    # parser.add_argument('--population_size', default=20000, type=int, help='population size')
    # parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    # parser.add_argument('--epochs', default=50, type=int, help='optimization epochs')
    parser.add_argument('--candidate', default=20, type=int, help='candidate')
    parser.add_argument('--final_selection', default=20, type=int, help='=candidate')
    parser.add_argument('--iterations', default=100, type=int, help='Number of iterations random transformations are applied.')

    # for test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_target', help='the only one target to test, or multiple targets separated by ,')
    

    args = parser.parse_args()

    main(args)