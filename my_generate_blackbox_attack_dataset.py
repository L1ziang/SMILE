#!/usr/bin/env python3
# coding=utf-8
import argparse
import glob
import os
import random
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from my_utils import crop_img, resize_img, normalize, create_folder, Tee
from my_target_models import get_model, get_input_resolution

random.seed(0)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666, help='set the seed')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout to mitigate overfitting')
    parser.add_argument('target_dataset', choices=['vggface', 'vggface2', 'CASIA'], help='use which target dataset')
    parser.add_argument('dataset', choices=['ffhq', 'celeba_partial256'], help='use which dataset')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    exp_name = os.path.join('blackbox_attack_data', args.target_dataset, args.arch_name, args.dataset)
    create_folder(exp_name)
    Tee(os.path.join(exp_name, 'output.log'), 'w')
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    net = get_model(args.arch_name, device, args.use_dropout)

    if args.arch_name == 'resnet50':
        resolution = get_input_resolution(args.arch_name)
    elif args.arch_name == 'inception_resnetv1_vggface2':
        resolution = get_input_resolution(args.arch_name)
    elif args.arch_name == 'inception_resnetv1_casia':
        resolution = get_input_resolution(args.arch_name)
    elif args.arch_name == 'mobilenet_v2':
        resolution = 224
    elif args.arch_name == 'efficientnet_b0':
        resolution = 256
    elif args.arch_name == 'efficientnet_b0_casia':
        resolution = 256
    elif args.arch_name == 'inception_v3':
        resolution = 342
    elif args.arch_name == 'swin_transformer':
        resolution = 260
    elif args.arch_name == 'vision_transformer':
        resolution = 224
    elif args.arch_name == 'vgg16':
        resolution = 224
    elif args.arch_name == 'vgg16bn':
        resolution = 224
    elif args.arch_name == 'sphere20a':
        resolution = get_input_resolution(args.arch_name)
    

    arch_name = args.arch_name

    if args.dataset == 'celeba_partial256':
        img_dir = './stylegan_sample_z_stylegan_celeba_partial256_0.7_8_25'
    elif args.dataset == 'ffhq':
        img_dir = './stylegan_sample_z_stylegan_ffhq256_0.7_8_25'
    imgs_files = sorted(glob.glob(os.path.join(img_dir, 'sample_*_img.pt')))

    assert len(imgs_files) > 0

    for img_gen_file in tqdm(imgs_files):
        save_filename = os.path.join(exp_name, os.path.basename(img_gen_file)[:-3]+'_logits.pt')
        fake = torch.load(img_gen_file).to(device)
        fake = crop_img(fake, arch_name)
        fake = normalize(resize_img(fake*255., resolution), args.arch_name)
        prediction = net(fake)
        if arch_name == 'sphere20a':
            prediction = prediction[0]
        torch.save(prediction, save_filename)


if __name__ == '__main__':
    main()
