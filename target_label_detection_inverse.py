import os
import copy
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import collections

import models

from helpers.loaders import get_data_transforms, get_dataloader, get_dataset, get_wm_transform, get_wm_path
from helpers.utils import get_trg_set, set_random_seed, set_up_logger
from trainer import test

from attacks.inversion_unlearning import inverse_class

def get_hyperparameters(dataset):
    args = {}
    args['NUM_INVERSION'] = 250
    args['epoches'] = 40
    args['dataset'] = dataset
    if dataset == 'cifar10':
        args['num_classes'] = 10
        args['mean'] = [0.4914, 0.4822, 0.4465]
        args['std'] = [0.2023, 0.1994, 0.2010]
        args['EPOCH_INVERSION'] = 1000
        args['xs_shape'] = (args['NUM_INVERSION'], 3, 32, 32)
    elif dataset == 'mnist':
        args['num_classes'] = 10
        args['mean'] = [0.0]
        args['std'] = [1.0]
        args['EPOCH_INVERSION'] = 400
        args['xs_shape'] = (args['NUM_INVERSION'], 1, 28, 28)
    elif dataset == 'cifar100':
        args['num_classes'] = 100
        args['mean'] = [0.4914, 0.4822, 0.4465]
        args['std'] = [0.2023, 0.1994, 0.2010]
        args['EPOCH_INVERSION'] = 500
        args['xs_shape'] = (args['NUM_INVERSION'], 3, 32, 32)
    else:
        raise NotImplementedError
    return args

import argparse
parser = argparse.ArgumentParser(description='Perform attacks on models.')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [default: cifar10]')
parser.add_argument('--smooth_k', default=50, type=int)
parser.add_argument('--method', default='WeaknessIntoStrength')
parser.add_argument('--wm_type', default='', type=str, help='e.g. content, noise, unrelated')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default='1', type=str, help='set gpu device (e.g. 0)')

args = parser.parse_args()
print(args)
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

set_random_seed(seed=args.seed)
cwd = os.getcwd()
set_up_logger(file=None)
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#####################################  Basic Settings  ###############################################

if args.dataset == 'cifar10':
    arch = 'resnet18'
    dataset = 'cifar10'
    num_classes = 10
elif args.dataset == 'cifar100':
    arch = 'resnet34'
    dataset = 'cifar100'
    num_classes = 100
else:
    raise NotImplementedError

method = args.method  #'EWE'
wm_type = args.wm_type  #'unrelated'

trg_set_size = 100
batch_size = 128
wm_batch_size = 32
data_ratio = 0.01667

method_path_name = method if (method != 'ProtectingIP') else method + wm_type
model_path = f"./checkpoint/lyf_{arch}_{dataset}_{method_path_name}_{trg_set_size}/best.pth"
print(model_path)

#####################################  Dataset & Model  ###############################################

# clean dataset
train_db_path = os.path.join(cwd, 'data')
test_db_path = os.path.join(cwd, 'data')
transform_train, transform_test = get_data_transforms(dataset)
train_set, test_set, valid_set = get_dataset(dataset, train_db_path, test_db_path, transform_train, transform_test, valid_size=data_ratio)
train_loader, test_loader, valid_loader = get_dataloader(train_set, test_set, batch_size, valid_set, shuffle=True)

# wm dataset
transform = get_wm_transform(method, dataset)
eps = 0.25 if method == 'FrontierStitching' else 0.1
wm_path = get_wm_path(method, dataset, wm_type=wm_type, model=arch, eps=eps, pattern_size='6')
print(f"wm_path {wm_path}")
trigger_set = get_trg_set(wm_path, 'labels.txt', trg_set_size, transform)
wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=wm_batch_size, shuffle=False)
print(trigger_set[0][1], trigger_set[1][1])

# load model
model = models.__dict__[arch](num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Test model
test_acc = test(model, None, test_loader, device, type='Test')
wm_acc = test(model, None, wm_loader, device, type='Watermark')
print(f"Test Acc {test_acc:.3f}, WM Acc {wm_acc:.3f}")

###################################  Smooth Inversion  #################################################

smooth_accs = [0.0] * num_classes
SMOOTH_K = int(args.smooth_k)

max_num = 100
if dataset == 'mnist' or dataset == 'fashion':
    sigma_input = 1.0
    sigma_param = 0.03
elif dataset == 'cifar10':
    sigma_input = 0.5
    sigma_param = 0.015
else:
    sigma_input = 0.5
    sigma_param = 0.015


# store_path = f"./reversed_detection/{base_store_name}.t"
base_store_name = f"reverse_lyf_{arch}_{method_path_name}nofix_{dataset}"
store_path = f"./reversed_800/{base_store_name}.t"

step = 1

if os.path.exists(store_path):
    class_xs_tensor = torch.load(store_path)
    print('found existing file at', store_path)
    print(f'loaded class_xs_tensor shape: {class_xs_tensor.shape}')
else:
    inverse_args = get_hyperparameters(dataset)
    # store_dir = f"./reversed_detection/{base_store_name}"
    store_dir = f"./reversed/{base_store_name}"
    os.makedirs(store_dir, exist_ok=True)
    class_xs = []
    for class_batch in range(0, num_classes, step):
        batch_store_path = f"{store_dir}/{class_batch}_{class_batch + step-1}.t"
        if os.path.exists(batch_store_path):
            class_xs_tensor_idx_cpu = torch.load(batch_store_path)
            print(f'loaded {batch_store_path}, class_xs_tensor shape: {class_xs_tensor_idx_cpu.shape}')
        else:
            class_xs_tensor_idx = inverse_class(model, device, inverse_args, list(range(class_batch, class_batch + step)))
            class_xs_tensor_idx_cpu = class_xs_tensor_idx.detach().cpu()
            del class_xs_tensor_idx
            torch.save(class_xs_tensor_idx_cpu, batch_store_path)
            print(f'reversed samples ({class_batch}_{class_batch + step-1}) are saved to {store_dir}/{class_batch}_{class_batch + step-1}.t')
        class_xs.append(class_xs_tensor_idx_cpu)
        torch.cuda.empty_cache()

    class_xs_tensor = torch.concat(class_xs)
    print(f"class_xs_tensor {class_xs_tensor.shape}")  

    torch.save(class_xs_tensor.detach(), store_path)
    print('reversed samples are saved to', store_path)

class_xs_tensor = class_xs_tensor.to(device)


with torch.no_grad():
    for i_smooth in range(SMOOTH_K):
        noises = {}
        for name, param in model.named_parameters():
            noise = torch.randn_like(param.data) * sigma_param
            noises[name] = noise
            param.data += noises[name]

        for i_class in range(num_classes):
            imgs = class_xs_tensor[250*i_class : 250*(i_class+1)]
            input_noise = torch.randn_like(imgs) * sigma_input
            outs = model(imgs + input_noise)
            _, predicted = torch.max(outs.data, dim=1)
            tmp_acc= (predicted == i_class).sum().item() / 250
            smooth_accs[i_class] += tmp_acc

        for name, param in model.named_parameters():
                param.data -= noises[name]

smooth_accs_dict = collections.defaultdict(float)

for i_class in range(num_classes):
    smooth_acc = smooth_accs[i_class] / SMOOTH_K
    smooth_accs_dict[i_class] = smooth_acc
    print(f"{i_class}, {smooth_acc:.3f}")

sort_smooth_list = sorted(smooth_accs_dict.items(), key=lambda x: x[1], reverse=True)
sort_smooth_dict = dict(sort_smooth_list)

smooth_store_path = f'./results_smooth/{dataset}_{arch}_{method_path_name}_{trg_set_size}'
with open(f'{smooth_store_path}.json', 'w') as f:
    json.dump(sort_smooth_dict, f)

print(f"max: {sort_smooth_list[0][0]}, second {sort_smooth_list[1][0]}")

if method == 'ProtectingIP':
    target_label_ori = trigger_set[1][1]
    print(f"Method {method} is fixed, Origin label {target_label_ori}")
    if sort_smooth_list[0][0] == target_label_ori:
        gap = sort_smooth_list[0][1] - sort_smooth_list[1][1]
        print(f"Detection Success! The margin is {gap}")
    else:
        ori_index = sort_smooth_list.index((target_label_ori, sort_smooth_dict[target_label_ori]))
        print(f"Detection failed, class {target_label_ori} with {sort_smooth_dict[target_label_ori]} is the {ori_index} large.")
        gap = sort_smooth_list[0][1] - sort_smooth_list[1][1]
        print(f"The margin is {gap}")
else:
    gap = sort_smooth_list[0][1] - sort_smooth_list[1][1]
    print(f"Method {method} is not fixed, the margin is {gap}")
