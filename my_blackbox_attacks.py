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

import vgg_m_face_bn_dag
import resnet50_scratch_dag
import vgg_face_dag
from facenet_pytorch import InceptionResnetV1
import net_sphere
from my_utils import normalize, clip_quantile_bound, create_folder, Tee, crop_and_resize, resize_img
from my_target_models import get_model, get_input_resolution

import nevergrad as ng
from tqdm import tqdm

from generator_RLBMI import Generator
from utils_RLBMI import * 
from SAC_RLBMI import Agent
from copy import deepcopy
from torchvision.utils import save_image
import pickle

from torch.utils.tensorboard import SummaryWriter

random.seed(0)

def get_input_resolution_self(arch_name):
    resolution = 224
    if arch_name.startswith('inception_resnetv1'):
        resolution = 160
    elif arch_name == 'sphere20a':
        resolution = (112, 96)
    elif arch_name.startswith('ccs19ami'):
        resolution = 64
        if 'rgb' not in arch_name:
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

class Sample:
    def __init__(self, value, fitness_score=-1):
        """
        value is a tensor
        """
        self.value = value
        self.fitness_score = fitness_score

class VectorizedPopulation_ours:
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

    def find_elite(self, index):
        self.fitness_scores, indices = torch.sort(self.fitness_scores, dim=0, descending=True)
        self.population = self.population[indices]
        return Sample(self.population[index].detach().clone(), self.fitness_scores[index].item())

    def visualize_imgs(self, filename, generate_images_func, k=8):
        ws = self.population[:k]
        out = generate_images_func(ws, raw_img=True)
        vutils.save_image(out, filename)

class VectorizedPopulation:
    def __init__(self, population, fitness_scores, mutation_prob, mutation_ce, apply_noise_func, clip_func, compute_fitness_func, clip_array_func):
        """
        population is a tensor with size N,512
        fitness_scores is a tensor with size N
        """
        self.population = population
        self.fitness_scores = fitness_scores
        self.mutation_prob = mutation_prob
        self.mutation_ce = mutation_ce
        self.apply_noise_func = apply_noise_func
        self.clip_func = clip_func
        self.compute_fitness_func = compute_fitness_func
        self.clip_array_func = clip_array_func

    def compute_fitness(self):
        bs = 50
        scores = []
        for i in range(0, len(self.population), bs):
            data = self.population[i:i+bs]
            scores.append(self.compute_fitness_func(data))
        self.fitness_scores = torch.cat(scores, dim=0)
        assert self.fitness_scores.ndim == 1 and self.fitness_scores.shape[0] == len(self.population)

    def find_elite(self, index=0):
        self.fitness_scores, indices = torch.sort(self.fitness_scores, dim=0, descending=True)
        self.population = self.population[indices]
        return Sample(self.population[index].detach().clone(), self.fitness_scores[index].item())

    def __get_parents(self, k):
        weights = F.softmax(self.fitness_scores, dim=0).tolist()
        parents_ind = random.choices(list(range(len(weights))), weights=weights, k=2*k)
        parents1_ind = parents_ind[:k]
        parents2_ind = parents_ind[k:]

        return parents1_ind, parents2_ind

    def __crossover(self, parents1_ind, parents2_ind):
        parents1_fitness_scores = self.fitness_scores[parents1_ind]
        parents2_fitness_scores = self.fitness_scores[parents2_ind]
        p = (parents1_fitness_scores / (parents1_fitness_scores + parents2_fitness_scores)).unsqueeze(1)  # size: N, 1
        parents1 = self.population[parents1_ind].detach().clone()  # size: N, 512
        parents2 = self.population[parents2_ind].detach().clone()  # size: N, 512
        mask = torch.rand_like(parents1)
        mask = (mask < p).float()
        return mask*parents1 + (1.-mask)*parents2

    def __mutate(self, children):
        mask = torch.rand_like(children)
        mask = (mask < self.mutation_prob).float()
        children = self.apply_noise_func(children, mask, self.mutation_ce)
        return self.clip_func(children)

    def produce_next_generation(self, elite):
        parents1_ind, parents2_ind = self.__get_parents(len(self.population)-1)
        children = self.__crossover(parents1_ind, parents2_ind)
        mutated_children = self.__mutate(children)
        self.population = torch.cat((elite.value.unsqueeze(0), mutated_children), dim=0)
        self.compute_fitness()

    def visualize_imgs(self, filename, generate_images_func, k=8):
        ws = self.population[:k]
        out = generate_images_func(ws, raw_img=True)
        vutils.save_image(out, filename)


def init_population(args):
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
    all_p_mins = all_p_means - args.p_std_ce * all_p_stds*args.x#2.5
    all_p_maxs = all_p_means + args.p_std_ce * all_p_stds*args.x#2.5
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
    if args.attack_mode == 'ours-current_maximum' \
    or args.attack_mode == 'ours-optimal_fit' \
    or args.attack_mode == 'ours-surrogate_model':
        return VectorizedPopulation_ours(population, fitness_scores, apply_noise_func, clip_func, args.compute_fitness_func, clip_array_func)

def init_population_mirror_b(args):
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
    all_ws = all_ws.to(args.device)
    
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
    all_logits = all_logits.to(args.device)
    all_prediction = F.log_softmax(all_logits, dim=1)[:, args.target]

    print('all_prediction.shape', all_prediction.shape, 'device', all_prediction.device)
    topk_conf, topk_ind = torch.topk(all_prediction, args.population_size, dim=0, largest=True, sorted=True)
    population = all_ws[topk_ind].detach().clone()
    fitness_scores = topk_conf

    return VectorizedPopulation(population, fitness_scores, args.mutation_prob, args.mutation_ce, apply_noise_func, clip_func, args.compute_fitness_func, clip_array_func)

def my_attack_current_maximum(args, generator, generate_images_func):

    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)
    writer = SummaryWriter(folder_path)
    Tee(os.path.join(folder_path, 'output.log'), 'w')

    test_model = get_model(args.test_arch_name, args.device)
    test_model = test_model.to(args.device)
    budget = args.budget
    T = 1
    strategy = args.strategy
    print(f"{args.target}")
    population = init_population(args)
    elite = population.find_elite(args.index)
    L = elite.value.cpu().numpy()
    parametrization = ng.p.Array(init=L).set_mutation(sigma=1)
    
    optimizer = ng.optimizers.registry[strategy](parametrization=parametrization, budget=budget)
    print(optimizer._info)

    img = generate_images_func(torch.Tensor(L).type(torch.float32).cuda().unsqueeze(0), raw_img=True)
    outputs = test_model(normalize(crop_and_resize(img, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
    logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]
    topk_conf, topk_class = torch.topk(outputs, 20, dim=1)
    print(f'{topk_class}')
    print(f'{logits_softmax}')

    score_0 = args.compute_fitness_func(elite.value.cuda().unsqueeze(0))
    img_0 = generate_images_func(elite.value.cuda().unsqueeze(0), raw_img=True)
    outputs_0 = test_model(normalize(crop_and_resize(img_0, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
    logits_softmax_0 = F.log_softmax(outputs_0, dim=1)[:, args.target]
    rank_of_label_0 = torch.sum(F.log_softmax(outputs_0, dim=1) > logits_softmax_0)

    writer.add_scalar('Target Score', score_0.item(), 0)
    writer.add_scalar('Evaluation Score', logits_softmax_0.item(), 0)
    writer.add_scalar('Rank of label', rank_of_label_0.item(), 0)

    res_original = Sample(elite.value.cpu().clone(), logits_softmax_0.item())
    torch.save(res_original, os.path.join(folder_path, 'original_w.pt'))

    pbar = tqdm(range(budget))
    for r in pbar:
        ng_data = [optimizer.ask() for _ in range(T)]
        for index in range(T):
            clipped_ng_data = population.clip_array_func(ng_data[index].value)
            ng_data[index].value[:] = clipped_ng_data
        
        score = [args.compute_fitness_func(torch.Tensor(ng_data[i].value).type(torch.float32).cuda().unsqueeze(0)) for i in range(T)]

        for z, l in zip(ng_data, score):
            optimizer.tell(z, l.item()*(-1.))
    
        recommendation = optimizer.provide_recommendation()
        recommendation = torch.Tensor(recommendation.value).type(torch.float32).cuda()

        recommendation = recommendation.unsqueeze(0)
        img = generate_images_func(recommendation, raw_img=True)

        outputs = test_model(normalize(crop_and_resize(img, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
        logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]

        rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

        if r % 10 == 0 or r == args.budget-1:
            writer.add_scalar('Target Score', score[0].item(), r+1)
            writer.add_scalar('Evaluation Score', logits_softmax.item(), r+1)
            writer.add_scalar('Rank of label', rank_of_label.item(), r+1)

            writer.add_image('Generated Image', img.squeeze(), global_step=r)
        
    res = Sample(recommendation.detach().clone(), logits_softmax.item())
    torch.save(res, os.path.join(folder_path, 'final_w.pt'))

    return

def my_attack_surrogate_model(args, generator, generate_images_func):

    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)
    writer = SummaryWriter(folder_path)
    Tee(os.path.join(folder_path, 'output.log'), 'w')

    test_model = get_model(args.test_arch_name, args.device)
    test_model = test_model.to(args.device)
    budget = args.budget
    T = 1
    strategy = args.strategy
    print(f"{args.target}")
    population = init_population(args)

    pre_path1 = './white_attack/' + 'ours-w'
    pre_path2 = args.EorOG + '-' + args.arch_name_target + '-' + args.dataset + '-' +  str(args.population_size) + '-' + str(args.epochs) + '-' + str(args.lr)+ '-' + str(args.arch_name_finetune) + '-' + str(args.finetune_mode) + '-' + str(args.num_experts)
    elite_path = os.path.join(pre_path1, pre_path2,'LOGS', str(args.target), str(args.index), 'final_w.pt')

    elite = torch.load(elite_path)
    elite.value = elite.value.squeeze()

    L = elite.value.cpu().numpy()

    parametrization = ng.p.Array(init=L).set_mutation(sigma=1)
    
    optimizer = ng.optimizers.registry[strategy](parametrization=parametrization, budget=budget)
    print(optimizer._info)

    img = generate_images_func(torch.Tensor(L).type(torch.float32).cuda().unsqueeze(0), raw_img=True)
    outputs = test_model(normalize(crop_and_resize(img, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
    if args.test_arch_name == 'sphere20a':
        outputs = outputs[0]
    logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]
    topk_conf, topk_class = torch.topk(outputs, 20, dim=1)
    print(f'{topk_class}')
    print(f'{logits_softmax}')

    score_0 = args.compute_fitness_func(elite.value.cuda().unsqueeze(0))
    img_0 = generate_images_func(elite.value.cuda().unsqueeze(0), raw_img=True)
    outputs_0 = test_model(normalize(crop_and_resize(img_0, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
    if args.test_arch_name == 'sphere20a':
        outputs_0 = outputs_0[0]
    logits_softmax_0 = F.log_softmax(outputs_0, dim=1)[:, args.target]
    rank_of_label_0 = torch.sum(F.log_softmax(outputs_0, dim=1) > logits_softmax_0)

    writer.add_scalar('Target Score', score_0.item(), 0)
    writer.add_scalar('Evaluation Score', logits_softmax_0.item(), 0)
    writer.add_scalar('Rank of label', rank_of_label_0.item(), 0)

    res_original = Sample(elite.value.cpu().clone(), logits_softmax_0.item())
    torch.save(res_original, os.path.join(folder_path, 'original_w.pt'))

    pbar = tqdm(range(budget))
    for r in pbar:
        ng_data = [optimizer.ask() for _ in range(T)]

        for index in range(T):
            clipped_ng_data = population.clip_array_func(ng_data[index].value)
            ng_data[index].value[:] = clipped_ng_data
        
        score = [args.compute_fitness_func(torch.Tensor(ng_data[i].value).type(torch.float32).cuda().unsqueeze(0)) for i in range(T)]

        for z, l in zip(ng_data, score):
            optimizer.tell(z, l.item()*(-1.))
    
        recommendation = optimizer.provide_recommendation()
        recommendation = torch.Tensor(recommendation.value).type(torch.float32).cuda()

        recommendation = recommendation.unsqueeze(0)
        img = generate_images_func(recommendation, raw_img=True)

        outputs = test_model(normalize(crop_and_resize(img, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
        if args.test_arch_name == 'sphere20a':
            outputs = outputs[0]
        logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]

        if r == 500 or r == 700 or r == 1000 or r == 1500:
            save_w = Sample(recommendation.detach().clone(), logits_softmax.item())
            save_w_name = str(r) + '_w.pt'
            torch.save(save_w, os.path.join(folder_path, save_w_name))

        rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

        if r % 10 == 0 or r == args.budget-1:
            writer.add_scalar('Target Score', score[0].item(), r+1)
            writer.add_scalar('Evaluation Score', logits_softmax.item(), r+1)
            writer.add_scalar('Rank of label', rank_of_label.item(), r+1)
            writer.add_image('Generated Image', img.squeeze(), global_step=r)
        
    res = Sample(recommendation.detach().clone(), logits_softmax.item())
    torch.save(res, os.path.join(folder_path, 'final_w.pt'))

    return

def my_attack_optimal_fit(args, generator, generate_images_func):

    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)
    writer = SummaryWriter(folder_path)
    Tee(os.path.join(folder_path, 'output.log'), 'w')

    test_model = get_model(args.test_arch_name, args.device)
    test_model = test_model.to(args.device)
    budget = args.budget
    T = 1
    strategy = args.strategy
    print(f"{args.target}")
    population = init_population(args)

    initial_params = [population.find_elite(I).value.cpu().numpy() for I in range(args.candidate_num)]
    second_best_index = 0 
    best_score = -float('inf')
    second_best_score = -float('inf') 

    for i, init in enumerate(initial_params):
        parametrization = ng.p.Array(init=init).set_mutation(sigma=1)
        optimizer = ng.optimizers.registry[args.strategy](parametrization=parametrization, budget=args.budget)
        
        for _ in range(args.early_stage):
            candidate = optimizer.ask()
            score = args.compute_fitness_func(torch.Tensor(candidate.value).type(torch.float32).cuda().unsqueeze(0))
            optimizer.tell(candidate, -score.item())
        
        recommendation = optimizer.provide_recommendation()
        final_score = args.compute_fitness_func(torch.Tensor(recommendation.value).type(torch.float32).cuda().unsqueeze(0)).item()
        
        if final_score > best_score:
            second_best_score = best_score
            second_best_index = best_index if 'best_index' in locals() else -1
            best_score = final_score
            best_index = i
        elif final_score > second_best_score:
            second_best_score = final_score
            second_best_index = i
        best_index = second_best_index

    best_init = initial_params[best_index]

    parametrization = ng.p.Array(init=best_init).set_mutation(sigma=1)
    optimizer = ng.optimizers.registry[strategy](parametrization=parametrization, budget=budget)
    print(optimizer._info)

    best_param_tensor = torch.Tensor(best_init)
    best_param_tensor = best_param_tensor.to(args.device)

    img = generate_images_func(torch.Tensor(best_init).type(torch.float32).cuda().unsqueeze(0), raw_img=True)
    outputs = test_model(normalize(crop_and_resize(img, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
    logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]
    topk_conf, topk_class = torch.topk(outputs, 20, dim=1)
    print(f'{topk_class}')
    print(f'{logits_softmax}')

    score_0 = args.compute_fitness_func(best_param_tensor.cuda().unsqueeze(0))
    img_0 = generate_images_func(best_param_tensor.cuda().unsqueeze(0), raw_img=True)
    outputs_0 = test_model(normalize(crop_and_resize(img_0, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
    logits_softmax_0 = F.log_softmax(outputs_0, dim=1)[:, args.target]
    rank_of_label_0 = torch.sum(F.log_softmax(outputs_0, dim=1) > logits_softmax_0)

    writer.add_scalar('Target Score', score_0.item(), 0)
    writer.add_scalar('Evaluation Score', logits_softmax_0.item(), 0)
    writer.add_scalar('Rank of label', rank_of_label_0.item(), 0)

    res_original = Sample(best_param_tensor.cpu().clone(), logits_softmax_0.item())
    torch.save(res_original, os.path.join(folder_path, 'original_w.pt'))

    pbar = tqdm(range(budget))
    for r in pbar:
        ng_data = [optimizer.ask() for _ in range(T)]
        for index in range(T):
            clipped_ng_data = population.clip_array_func(ng_data[index].value)
            ng_data[index].value[:] = clipped_ng_data
        
        score = [args.compute_fitness_func(torch.Tensor(ng_data[i].value).type(torch.float32).cuda().unsqueeze(0)) for i in range(T)]

        for z, l in zip(ng_data, score):
            optimizer.tell(z, l.item()*(-1.))
    
        recommendation = optimizer.provide_recommendation()
        recommendation = torch.Tensor(recommendation.value).type(torch.float32).cuda()

        recommendation = recommendation.unsqueeze(0)
        img = generate_images_func(recommendation, raw_img=True)

        outputs = test_model(normalize(crop_and_resize(img, args.test_arch_name, args.test_resolution)*255., args.test_arch_name))
        logits_softmax = F.log_softmax(outputs, dim=1)[:, args.target]

        rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

        if r % 100 == 0 or r == args.budget-1:
            writer.add_scalar('Target Score', score[0].item(), r+1)
            writer.add_scalar('Evaluation Score', logits_softmax.item(), r+1)
            writer.add_scalar('Rank of label', rank_of_label.item(), r+1)

            writer.add_image('Generated Image', img.squeeze(), global_step=r)
        
    res = Sample(recommendation.detach().clone(), logits_softmax.item())
    torch.save(res, os.path.join(folder_path, 'final_w.pt'))

    return

def my_attack_mirror_b(args, generator, generate_images_func):

    folder_path = args.exp_name + '/LOGS/' + str(args.target) +'/'+ str(args.index)
    os.makedirs(folder_path, exist_ok=True)
    population = init_population_mirror_b(args)
    generations = args.generations
    for gen in range(generations):
        elite = population.find_elite()
        print(f'elite at {gen}-th generation: {elite.fitness_score}')
        population.visualize_imgs(os.path.join(folder_path, f'{gen}.png'), generate_images_func)

        if elite.fitness_score >= math.log(args.min_score): 
            torch.save(elite, os.path.join(folder_path, 'final_w.pt'))
            return
        population.produce_next_generation(elite)

    elite = population.find_elite(index=0)
    torch.save(elite, os.path.join(folder_path, 'final_w.pt'))

    return

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
        elif args.test_arch_name == 'efficientnet_b0_casia':
            base_dir = '/root/autodl-tmp/CASIA-WebFace_eval/eval/efficientnet_b0_casia'
        
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
    print(test_true_list)

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

features = {}

def main(args):

    torch.backends.cudnn.benchmark = True
    print(args)
    print(datetime.now())

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f'using device: {device}')

    net = get_model(args.arch_name_target, device)
    net = net.to(device)
    net.eval()

    # VGGFace
    if args.arch_name_target == 'vgg16bn':
        args.test_arch_name = 'vgg16'
    elif args.arch_name_target == 'vgg16':
        args.test_arch_name = 'vgg16bn'

    # VGGFace2
    elif args.arch_name_target == 'resnet50':
        args.test_arch_name = 'inception_resnetv1_vggface2'
    elif args.arch_name_target == 'inception_resnetv1_vggface2':
        args.test_arch_name = 'resnet50'
    
    elif args.arch_name_target == 'mobilenet_v2':
        args.test_arch_name = 'resnet50'
    elif args.arch_name_target == 'efficientnet_b0':
        args.test_arch_name = 'resnet50'
    elif args.arch_name_target == 'inception_v3':
        args.test_arch_name = 'resnet50'
    elif args.arch_name_target == 'swin_transformer':
        args.test_arch_name = 'resnet50'
    elif args.arch_name_target == 'vision_transformer':
        args.test_arch_name = 'resnet50'

    elif args.arch_name_target == 'inception_resnetv1_casia':
        args.test_arch_name = 'efficientnet_b0_casia'
    elif args.arch_name_target == 'efficientnet_b0_casia':
        args.test_arch_name = 'inception_resnetv1_casia'
    elif args.arch_name_target == 'sphere20a':
        args.test_arch_name = 'inception_resnetv1_casia'
    else:
        raise AssertionError('wrong arch_name')
    
    args.resolution = get_input_resolution_self(args.arch_name_target)
    args.test_resolution = get_input_resolution_self(args.test_arch_name)

    if args.attack_mode == 'ours-current_maximum' \
    or args.attack_mode == 'ours-optimal_fit' \
    or args.attack_mode == 'ours-surrogate_model' \
    or args.attack_mode == 'Mirror-b':
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

        def generate_images_func(w, raw_img=False):
            assert w.ndim == 2
            if raw_img:
                return generator(w.to(device))
            img = crop_and_resize(generator(w.to(device)), args.arch_name_target, args.resolution)
            return img

        def compute_fitness_func(w):
            img = generate_images_func(w)
            assert img.ndim == 4
            if args.arch_name_target == 'sphere20a':
                pred = F.log_softmax(net(normalize(img*255., args.arch_name_target))[0], dim=1)
            else:
                pred = F.log_softmax(net(normalize(img*255., args.arch_name_target)), dim=1)

            score = pred[:, args.target]
            return score 
        args.compute_fitness_func = compute_fitness_func

        path_exp = './blackbox_attack/' + args.attack_mode
        if args.attack_mode == 'ours-current_maximum':
            exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' + args.strategy + '-' + str(args.budget) + '-' + str(args.population_size) # 根据attack mode创建
        elif args.attack_mode == 'ours-optimal_fit':
            exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' + args.strategy + '-' + str(args.budget) + '-' + str(args.population_size) + '-' + str(args.early_stage)+ '-' + str(args.candidate_num)# 根据attack mode创建
        elif args.attack_mode == 'ours-surrogate_model':
            exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' + args.strategy + '-' + str(args.budget) + '-' + str(args.population_size) + '-' + args.EorOG + '-' + str(args.epochs) + '-' + str(args.lr)+ '-' + str(args.arch_name_finetune) + '-' + str(args.finetune_mode) + '-' + str(args.num_experts)# 根据attack mode创建
        elif args.attack_mode == 'Mirror-b':
            exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' +  str(args.population_size) + '-' + str(args.generations)

        args.exp_name = os.path.join(path_exp, exp_name_tmp)
        create_folder(args.exp_name)

        if args.test_only:
            if args.attack_mode == 'ours-current_maximum' \
            or args.attack_mode == 'ours-optimal_fit' \
            or args.attack_mode == 'ours-surrogate_model':
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
            
            if args.attack_mode == 'Mirror-b':
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

                elif args.test_arch_name == 'efficientnet_b0_casia':
                    def hook_fn(module, input, output):
                        features['fs'] = output
                    layer_to_hook = test_net.avgpool
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


        if args.attack_mode == 'ours-current_maximum':
            args.index = 0
            my_attack_current_maximum(args, generator, generate_images_func)
        elif args.attack_mode == 'ours-optimal_fit':
            args.index = 0
            my_attack_optimal_fit(args, generator, generate_images_func)
        elif args.attack_mode == 'ours-surrogate_model':
            args.index = 0
            my_attack_surrogate_model(args, generator, generate_images_func)
        elif args.attack_mode == 'Mirror-b':
            args.index = 0
            my_attack_mirror_b(args, generator, generate_images_func)


    if args.attack_mode == 'RLB-MI':
        G = Generator(args.z_dim)
        G = nn.DataParallel(G).to(args.device)
        if args.dataset == 'celeba_RLBMI':
            ckp_G = torch.load('./checkpoints/celeba_G.tar')['state_dict']
        elif args.dataset == 'ffhq_RLBMI':
            ckp_G = torch.load('./checkpoints/ffhq_G.tar')['state_dict']
        load_my_state_dict(G, ckp_G)
        G.eval()

        path_exp = './blackbox_attack/' + args.attack_mode
        exp_name_tmp = args.arch_name_target + '-' + args.dataset + '-' + str(args.max_episodes)
        args.exp_name = os.path.join(path_exp, exp_name_tmp)
        create_folder(args.exp_name)

        if args.test_only:
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
            for t in targets:
                folder_path = args.exp_name + '/LOGS/' + str(t)
                final_sample = torch.load(os.path.join(folder_path, 'final_z.pt'))
                w = final_sample#.value
                ws.append(w.to(device))
                score_path = os.path.join(folder_path, 'score_data.pkl')
                with open(score_path, 'rb') as file:
                    score= pickle.load(file)
                all_confs.append(score)
            ws = torch.stack(ws, dim=0)
            ws = ws.squeeze(dim=1)
            imgs = G(ws.to(device))
            test_true_list, target_conf_test = compute_conf(args, args.exp_name, test_net, args.test_arch_name, args.test_resolution, targets, imgs)

            imgs = resize_img(imgs, 256)
            imgs, imgs_origanal = add_conf_to_tensors_custom(imgs, target_conf_test, test_true_list=test_true_list, testornot=True)
            create_folder('./tmp')
            formatted_exp_name = args.exp_name.replace('./', '')
            path_png = './tmp/' + formatted_exp_name + '.png'
            path_png_og = './tmp/' + formatted_exp_name + '_og.png'
            vutils.save_image(imgs, path_png, nrow=1)
            vutils.save_image(imgs_origanal, path_png_og, nrow=1)
        
            return

        agent = Agent(state_size=args.z_dim, action_size=args.z_dim, random_seed=666, hidden_size=256, action_prior="uniform")

        for i_episode in tqdm(range(1, args.max_episodes + 1), desc="Episodes"):
            y = torch.tensor([args.target]).to(args.device)

            z = torch.randn(1, args.z_dim).to(args.device)
            state = deepcopy(z.cpu().numpy())
            for t in range(args.max_step):
                action = agent.act(state)
                z = args.alpha * z + (1 - args.alpha) * action.clone().detach().reshape((1, len(action))).to(args.device)
                next_state = deepcopy(z.cpu().numpy())
                state_image = G(z).detach()
                action_image = G(action.clone().detach().reshape((1, len(action))).to(args.device)).detach()

                state_image = crop_and_resize(state_image, args.arch_name_target, args.resolution)
                state_output = net(normalize(state_image*255., args.arch_name_target))
                action_image = crop_and_resize(action_image, args.arch_name_target, args.resolution)
                action_output = net(normalize(action_image*255., args.arch_name_target))

                score1 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(state_output, dim=-1)).data, 1, y))))
                score2 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_output, dim=-1)).data, 1, y))))
                score3 = math.log(max(1e-7, float(torch.index_select(F.softmax(state_output, dim=-1).data, 1, y)) - float(torch.max(torch.cat((F.softmax(state_output, dim=-1)[0,:y],F.softmax(state_output, dim=-1)[0,y+1:])), dim=-1)[0])))
                reward = 2 * score1 + 2 * score2 + 8 * score3

                if t == args.max_step - 1 :
                    done = True
                else :
                    done = False

                agent.step(state, action, reward, next_state, done, t)
                state = next_state
            
        test_images = []
        test_scores = []
        for i in range(1):
            with torch.no_grad():
                z_test = torch.randn(1, args.z_dim).to(args.device)
                for t in range(args.max_step):
                    state_test = z_test.cpu().numpy()
                    action_test = agent.act(state_test)
                    z_test = args.alpha * z_test + (1 - args.alpha) * action_test.clone().detach().reshape((1, len(action_test))).to(args.device)
                test_image = G(z_test).detach()
                test_images.append(test_image.cpu())
                test_image = crop_and_resize(test_image, args.arch_name_target, args.resolution)
                test_output = net(normalize(test_image*255., args.arch_name_target))

                test_score = float(torch.mean(torch.diag(torch.index_select(F.softmax(test_output, dim=-1).data, 1, y))))
            test_scores.append(test_score)

        folder_path = args.exp_name + '/LOGS/' + str(args.target)
        os.makedirs(folder_path, exist_ok=True)
        image_path = folder_path + '/' + 'result.png'
        save_image(test_image, image_path)

        print(f'final confidence: {test_score}')
        save_score_path = os.path.join(folder_path, 'score_data.pkl')
        with open(save_score_path, 'wb') as file:
            pickle.dump(test_score, file)
        torch.save(z_test, os.path.join(folder_path, 'final_z.pt'))

    exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_mode', type=str, choices=['AMI', 'RLB-MI', 'Mirror-b', 'ours-current_maximum', 'ours-optimal_fit', 'ours-surrogate_model'], help='black-box attack mode')
    parser.add_argument('--seed', type=int, default=666, help="random seed")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default=None, help='name')

    parser.add_argument('--target_dataset', default='CASIA', type=str, choices=['vggface2', 'vggface', 'CASIA'], help='test dataset')
    parser.add_argument('--dataset', default='celeba_partial256', type=str, choices=['ffhq', 'celeba_partial256', 'celeba_RLBMI', 'ffhq_RLBMI'], help='stylegan dataset')
    parser.add_argument('--arch_name_target', default='inception_resnetv1_casia', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name from torchvision or resnet50v15')

    parser.add_argument('--bs', default=1000, type=int, help='batch size')

    parser.add_argument('--target', default=8, type=int, help='the target label')

    parser.add_argument('--p_std_ce', type=float, default=1., help='set the bound for p_space_bound mean+-x*std')

    # for SMILE
    parser.add_argument('--index', default=0, type=int, help='Top-k index of initialization')
    parser.add_argument('--budget', default=2000, type=int, help='budget')
    parser.add_argument('--strategy', default='NGOpt', type=str, choices=['MetaModel', 'RealSpacePSO', 'PSO', 'ASCMADEthird', 'GeneticDE', 'NGO', 'NGOpt'], help='strategy')
    # parser.add_argument('--population_size', default=10000, type=int, help='population size')

    # for SMILE-optimal_fit
    parser.add_argument('--early_stage', type=int, default=50)
    parser.add_argument('--candidate_num', type=int, default=10)

    # for SMILE-surrogate_model
    parser.add_argument('--EorOG', default='SMILE', type=str, choices=['E', 'E_ce', 'SMILE', 'OG'], help='mode')
    parser.add_argument('--arch_name_finetune', default='inception_resnetv1_vggface2', type=str, choices=['resnet50', 'inception_resnetv1_vggface2', 'mobilenet_v2', 'efficientnet_b0', 'inception_v3', 'swin_transformer', 'vgg16', 'vision_transformer', 'vgg16bn', 'inception_resnetv1_casia', 'sphere20a', 'efficientnet_b0_casia'], help='model name from torchvision or resnet50v15')
    parser.add_argument('--finetune_mode', default='CASIA->vggface2', type=str, choices=['vggface->vggface2', 'vggface->CASIA', 'vggface2->CASIA', 'CASIA->vggface2'], help='mode')
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--population_size', default=10000, type=int, help='population size')
    parser.add_argument('--x', type=float, default=1.7, help='var')

    # parser.add_argument('--population_size', default=10000, type=int, help='population size')
    parser.add_argument('--epochs', default=1000, type=int, help='optimization epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for optimization')
    
    # for RLB-MI
    parser.add_argument("--max_episodes", type=int, default=40_000)
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--z_dim", type=int, default=100)

    # for Mirror-b
    # parser.add_argument('--population_size', default=20000, type=int, help='population size')
    parser.add_argument('--generations', default=10, type=int, help='total generations')
    parser.add_argument('--mutation_prob', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--mutation_ce', type=float, default=0.1, help='mutation coefficient')
    parser.add_argument('--min_score', type=float, default=0.95, help='once reaching the score, terminate the attack')

    # for test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_target', help='the only one target to test, or multiple targets separated by ,')

    
    args = parser.parse_args()

    if args.attack_mode == 'RLB-MI':
        main(args)
    else:
        with torch.no_grad():
            main(args)
