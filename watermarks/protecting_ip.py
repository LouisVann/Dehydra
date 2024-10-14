"""Protecting Intellectual Property of Deep Neural Networks with Watermarking (Zhang et al., 2018)

- different wm_types: ('content', 'unrelated', 'noise')"""

from watermarks.base import WmMethod

import os
import logging
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import save_triggerset, find_tolerance, get_trg_set
from helpers.loaders import get_data_transforms, get_wm_transform
from helpers.transforms import EmbedText, EmbedBlock

from trainer import test, train, train_on_augmented


class ProtectingIP(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'protecting_ip')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.wm_type = args.wm_type  # content, unrelated, noise
        self.p = None

    '''
    modify: we need to make sure the image (either from mnist or cifar10)
    ## AFTER applying trigger, BEFORE normlizing ##
    is clipped into [0, 1]
    '''
    def gen_watermarks(self, device): #, seed = 0):
        logging.info('Generating watermarks. Type = ' + self.wm_type)
        datasets_dict = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100, 'mnist': datasets.MNIST, 'fashion': datasets.FashionMNIST}

        # in original: one trigger label for ALL trigger images. went with label_watermark=lambda w, x: (x + 1) % 10
        # trigger_lbl = 1  # "airplane"

        if self.wm_type == 'content':
            wm_dataset = self.dataset

            if self.dataset == "cifar10" or self.dataset == "cifar100":
                transform_watermarked = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    EmbedText("TEST", (0, 22), 0.5),
                    transforms.Lambda(lambda x: torch.clamp(x, min=0., max=1.)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif self.dataset == "mnist" or self.dataset == 'fashion':
                transform_watermarked = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    EmbedText("TEST", (0, 18), 0.5),
                    transforms.Lambda(lambda x: torch.clamp(x, min=0., max=1.)),
                ])

        elif self.wm_type == 'unrelated':
            if self.dataset == 'mnist' or self.dataset == 'fashion':
                wm_dataset = 'cifar10'
                # normalize like cifar10, crop like mnist
                transform_watermarked = transforms.Compose([
                    transforms.Resize(28),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.clamp(x, min=0., max=1.)),
                ])
            elif self.dataset == 'cifar10' or self.dataset == "cifar100":
                wm_dataset = 'mnist'
                # crop like cifar10, make rgb
                transform_watermarked = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Lambda(lambda x: torch.clamp(x, min=0., max=1.)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        elif self.wm_type == 'noise':
            wm_dataset = self.dataset
            # add gaussian noise to trg images
            transform_train, _ = get_data_transforms(self.dataset)

            if self.dataset == 'mnist' or self.dataset == 'fashion':
                transform_watermarked = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
                    transforms.Lambda(lambda x: torch.clamp(x, min=0., max=1.)),
                ])

            elif self.dataset == 'cifar10'or self.dataset == "cifar100":
                transform_watermarked = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + 0.4 * torch.randn_like(x)),
                    transforms.Lambda(lambda x: torch.clamp(x, min=0., max=1.)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        

        wm_set = datasets_dict[wm_dataset](root='./data', train=True, download=False, transform=transform_watermarked)
        # modify: Following Zhang, we collect watermarks samples belonging to source class, and assign labels to target class
        source_class = 4
        target_class = 6 #random.randint(0,10)
        # print(wm_set[0][0][0])
        idx_list = []
        for i in random.sample(range(len(wm_set)), len(wm_set)):  # iterate randomly
            img, lbl = wm_set[i]
            if lbl != source_class:
                continue
            img_max = img.max()
            img_min = img.min()
            # img = img.to(device)
            trg_lbl = torch.tensor(target_class)
            # trg_lbl = (lbl + 1) % self.num_classes  # set trigger labels label_watermark=lambda w, x: (x + 1) % 10
            self.trigger_set.append((img, trg_lbl))
            idx_list.append(i)

            if len(self.trigger_set) == self.size:
                break  # break for loop when trigger set has final size

        if self.save_wm:  # save to pre-defined path
            save_triggerset(self.trigger_set, self.path, self.dataset, self.wm_type)#  + f"_{seed}")
        return self.trigger_set


    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        # self.gen_watermarks(device) # old
        transform = get_wm_transform('ProtectingIP', self.dataset)

        # logging.info('Loading WM dataset.')
        self.trigger_set = get_trg_set(os.path.join(self.path, self.wm_type, self.dataset), 'labels.txt', self.size, transform=transform)
        self.loader()
        # self.wm_loader = torch.utils.data.DataLoader(self.trigger_set, batch_size=self.wm_batch_size, shuffle=True)

        logging.info("Begin embedding.")
        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, self.wm_loader, save_dir, self.save_model, self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch
    