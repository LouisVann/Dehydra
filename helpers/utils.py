"""Some helper functions for PyTorch, including:
    - count_parameters: calculate parameters of network and display as a pretty table.
    - progress_bar: progress bar mimic xlua.progress.
"""
import csv
import os
import sys
import re
import time
import logging
import pickle
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset

from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt

from helpers.image_folder_custom_class import ImageFolderCustomClass

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def progress_bar(current, total, msg=None):
    """ creates progress bar for training"""
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    l = list()
    l.append('  Step: %s' % format_time(step_time))
    l.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        l.append(' | ' + msg)

    msg = ''.join(l)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def adjust_learning_rate(init_lr, optimizer, epoch, lradj):
    """Sets the learning rate to the initial LR decayed by 10 every /epoch/ epochs"""
    lr = init_lr * (0.1 ** (epoch // lradj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# finds a value for theta (maximum number of errors tolerated for verification) (frontier-stitching)
def find_tolerance(key_length, threshold):
    theta = 0
    factor = 2 ** (-key_length)
    s = 0
    # while True:
    #     # for z in range(theta + 1):
    #     s += binomial(key_length, theta)
    #     if factor * s >= threshold:
    #         return theta
    #     theta += 1

    while factor * s < threshold:
        s += binomial(key_length, theta)
        theta += 1

    return theta


# (frontier-stitching)
def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b //= t+1
        n -= 1
    return b



def set_up_logger(file):
    # create custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # format for our loglines
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if file is not None:
        # setup file logging as well
        fh = logging.FileHandler(file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_triggerset(trigger_set, path, dataset, wm_type=None):
    logging.info('Saving watermarks.')
    os.makedirs(path, exist_ok=True)

    if wm_type:
        path = os.path.join(path, wm_type) 
    path = os.path.join(path, dataset)

    os.makedirs(path, exist_ok=True)

    pics_d = os.path.join(path, 'pics')
    labels_f = os.path.join(path, 'labels.txt')

    if not os.path.isdir(pics_d):
        os.mkdir(pics_d)

    # maybe: https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty

    if os.path.exists(labels_f):
        os.remove(labels_f)

    for idx, (img, lbl) in enumerate(trigger_set):
        save_image(img, os.path.join(pics_d, str(idx+1) + '.jpg'))
        with open(labels_f, 'a') as f:
            int_label = int(lbl.item()) if torch.is_tensor(lbl) else int(lbl)
            # if torch.is_tensor(lbl):
            #     f.write(str() + '\n')
            # else:
            #     f.write(str(lbl) + '\n')
            f.write(str(int_label) + '\n')

# WM for embedded systems
def add_watermark(tensor, watermark):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m in zip(tensor, watermark):
        t.add_(m)
    return tensor

def get_trg_set(path, labels, size, transform=None):
    wm_set = ImageFolderCustomClass(path, transform)
    img_nlbl = list()
    wm_targets = np.loadtxt(os.path.join(path, labels))
    for idx, (path, target) in enumerate(wm_set.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
        if idx == (size - 1):
            break

    wm_set.imgs = img_nlbl
        
    return wm_set


def save_results(csv_args, csv_file):
    logging.info("Saving results.")
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(csv_args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_up_optim_sched(net, lr, opti, sched, lradj=None, T_max=90, method = None):

    if opti == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
    elif opti == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        raise NotImplementedError('optimizer')

    if sched == 'MultiStepLR':
        assert lradj is not None
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180], gamma=lradj)
    elif sched == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-4)
    else:
        raise NotImplementedError('scheduler')

    return optimizer, scheduler

def wm_set_up_optim_sched(net, lr, opti, sched, lradj=None, T_max=90, method = None):

    if opti == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opti == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        raise NotImplementedError('optimizer')
    if method == 'Piracy':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-5)

    if sched == 'MultiStepLR':
        assert lradj is not None
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180], gamma=lradj)
    elif sched == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-4)
    else:
        raise NotImplementedError('scheduler')

    return optimizer, scheduler



def zip_checkpoint_dir(save_dir, save_model):
    dir = os.path.join(save_dir, save_model + '_duringtraining')

    # zip dir
    shutil.make_archive(dir, 'zip', dir)

def get_max_index(data_dir, suffix):
    """ Lists all files from a folder and checks the largest integer prefix for a filename (in snake_case) that
    contains the suffix
    """
    index = 0
    for filename in os.listdir(data_dir):
        if suffix in filename:
            index = int(filename.split("_")[0]) + 1 if int(filename.split("_")[0]) >= index else index
    return str(index)

def collect_n_samples(n, data_loader, class_label = None, has_labels = False, reduce_labels = False):
    # need tensor
    x_samples, y_samples = torch.empty(0),torch.empty(0)
    # print(type(x_samples).__name__)
    if has_labels:
        for (x, y) in data_loader:
            
            # if len(x_samples) >= n:
            if x_samples.shape[0] >= n:
                break
            # Reduce soft labels.
            y_full = y.clone()
            if y.dim() > 1:
                y = y.argmax(dim=1)

            # Compute indices of samples we want to keep.
            idx = np.arange(x.shape[0])
            if class_label:
                idx, = np.where(y == class_label)

            if len(idx) > 0:
                # x_samples.extend(x[idx].detach().cpu().numpy())
                x_samples = torch.cat((x[idx].detach().cpu(), x_samples),dim=0)
                if reduce_labels:
                    # y_samples.extend(y[idx].detach().cpu().numpy())
                    y_samples = torch.cat((y[idx].detach().cpu(), y_samples),dim=0)
                else:
                    # y_samples.extend(y_full[idx].detach().cpu().numpy())
                    y_samples = torch.cat((y_full[idx].detach().cpu(), y_samples),dim=0)

        if n == np.inf:
            return x_samples, y_samples

        # if len(x_samples) < n:
        if x_samples.shape[0] < n:
            print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
        return x_samples[:n], y_samples[:n]
    
    else:   # No labels.
        for x,_ in data_loader:
            # print(x, type(x))
            # x_samples.extend(x.detach().cpu())
            x_samples = torch.cat((x.detach().cpu(), x_samples), dim=0)
            # if len(x_samples) >= n:
            if x_samples.shape[0] >= n:
                break

        # if len(x_samples) < n:
        if x_samples.shape[0] < n:
            print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
        # print(type(x_samples).__name__)
        return x_samples[:n]
    
def check_data_ratio(args):
    if args.attack_type in ['pruning', 'fine-pruning', 'fine-tuning']:
        return 
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_size = 50000
        test_size = 10000
        data_ratio = 0.01667
    elif args.dataset == 'mnist':
        train_size = 60000
        test_size = 10000
        data_ratio = 0.0142857
    else:
        raise NotImplementedError
    
    if args.data_ratio != data_ratio:
        print(f"[Attention] for dataset {args.dataset} and attack {args.attack_type}, the data_ratio should be {data_ratio} not {args.data_ratio}.")

    