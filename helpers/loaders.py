import os

import numpy as np
import torch
import random
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, random_split


def get_data_transforms(datatype):
    if datatype.lower() == 'cifar10' or datatype.lower() == 'cifar100' or datatype.lower() == 'cinic10' or datatype.lower() == 'cinic10-imagenet':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([ 
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif datatype.lower() == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif datatype.lower() == 'mnist' or datatype.lower() == 'emnist' or datatype.lower() == 'fashion':
        transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])
    elif datatype.lower() == 'svhn':
        transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])
    else:
        raise NotImplementedError

    return transform_train, transform_test


def get_wm_transform(method, dataset):
    if method == 'WeaknessIntoStrength':
        if dataset == "cifar10" or dataset == "cifar100":
            transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist" or dataset == 'fashion':
            transform = transforms.Compose([
                transforms.CenterCrop(28),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

    elif method == 'ProtectingIP':
        if dataset == 'mnist' or dataset == 'fashion':
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.ToTensor(),
                                            ])
        elif dataset == 'cifar10' or dataset == "cifar100":
            transform = transforms.ToTensor()
    else:
        if dataset == "cifar10" or dataset == "cifar100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "mnist" or dataset == 'fashion':
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

    return transform


def get_data_subset(train_set, test_set, testquot=None, size_train=None, size_test=None):
    # check this out: traindata_split = torch.utils.data.random_split(traindata,
    #                                           [int(traindata.data.shape[0] / partitions) for _ in range(partitions)])
    if testquot:
        size_train = len(train_set) / testquot
        size_test = len(test_set) / testquot

    sub_train = random.sample(range(len(train_set)), int(size_train))
    sub_test = random.sample(range(len(test_set)), int(size_test))

    train_set = torch.utils.data.Subset(train_set, sub_train)
    test_set = torch.utils.data.Subset(test_set, sub_test)

    return train_set, test_set


def get_dataset(datatype, train_db_path, test_db_path, transform_train, transform_test, valid_size=None,
                testquot=None, size_train=None, size_test=None):
    logging.info('Loading dataset. Dataset: ' + datatype)
    datasets_dict = {'cifar10': datasets.CIFAR10,
                     'cifar100': datasets.CIFAR100,
                     'mnist': datasets.MNIST,
                     'stl10': datasets.STL10,
                     'svhn': datasets.SVHN,
                     'emnist': datasets.EMNIST,
                     'fashion': datasets.FashionMNIST}

    # Datasets
    if datatype == 'svhn' or datatype == 'stl10':
        train_set = datasets_dict[datatype](root=train_db_path,
                                            split='train', transform=transform_train,
                                            download=False)
        test_set = datasets_dict[datatype](root=test_db_path,
                                           split='test', transform=transform_test,
                                           download=False)
    elif datatype == 'emnist':
        train_set = datasets_dict[datatype](root=train_db_path, split='digits', train=True, download=False,
                                            transform=transform_train)

        test_set = datasets_dict[datatype](root=train_db_path, split='digits', train=False, download=False,
                                           transform=transform_test)
        # print(f"emnist {len(test_set)}")
    elif datatype == 'cinic10':
        cinic_directory = os.path.join(train_db_path, 'cinic-10')
        train_set = datasets.ImageFolder(os.path.join(cinic_directory, 'train'),
                                         transform=transform_train)
        test_set = datasets.ImageFolder(os.path.join(cinic_directory, 'test'),
                                        transform=transform_test)
    elif datatype == 'cinic10-imagenet':
        cinic_directory = os.path.join(train_db_path, 'cinic-10-imagenet')
        train_set = datasets.ImageFolder(os.path.join(cinic_directory, 'train'),
                                         transform=transform_train)
        test_set = datasets.ImageFolder(os.path.join(cinic_directory, 'test'),
                                        transform=transform_test)
    else:
        train_set = datasets_dict[datatype](root=train_db_path,
                                            train=True, download=False,
                                            transform=transform_train)
        test_set = datasets_dict[datatype](root=test_db_path,
                                           train=False, download=False,
                                           transform=transform_test)
        # print(f"fashion {len(train_set)} {len(test_set)}")

    # using only a subset of dataset - for testing reasons
    if testquot:
        logging.info("Using 1/%d subset of %r." % (testquot, datatype))
        train_set, test_set = get_data_subset(train_set, test_set, testquot)
    if size_train:
        logging.info("Using a subset of %r of size (%d, %d)." % (datatype, size_train, size_test))
        train_set, test_set = get_data_subset(train_set, test_set, testquot, size_train, size_test)

    if valid_size:
        n = len(train_set) + len(test_set)  # lyf changed here
        n_valid = int(valid_size * n)
        train_set, valid_set = random_split(train_set, [len(train_set) - n_valid, n_valid])
    else:
        valid_set = None

    return train_set, test_set, valid_set


def get_dataloader(train_set, test_set, batch_size, valid_set=None, shuffle=True, drop_last = False):
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0,
                                               shuffle=shuffle, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0,
                                              shuffle=False, drop_last=drop_last)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=0,
                                               shuffle=False, drop_last=drop_last)

    logging.info('Size of training set: %d, size of testing set: %d' % (len(train_set), len(test_set)))

    return train_loader, test_loader, valid_loader


def get_wm_path(method, dataset, wm_type=None, model=None, eps=None, pattern_size=None):
    if method == 'ProtectingIP':
        return os.path.join('data', 'trigger_set', 'protecting_ip', wm_type, dataset)

    elif method == 'FrontierStitching':
        return os.path.join('data', 'trigger_set', 'frontier_stitching', model, str(float(eps)), dataset)

    elif method == 'WeaknessIntoStrength':
        return os.path.join('data', 'trigger_set', 'weakness_into_strength')

    elif method == 'ExponentialWeighting':
        return os.path.join('data', 'trigger_set', 'exponential_weighting_new', dataset)

    elif method == 'WMEmbeddedSystems':
        return os.path.join('data', 'trigger_set', 'wm_embedded_systems', 'num_bits' + str(pattern_size), 'strength' + str(eps), dataset)
