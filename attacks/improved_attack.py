from helpers.loaders import get_data_transforms, get_dataset, get_dataloader
from trainer import train, test
from utils import finetune
import copy

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_hyperparameters(dataset):
    args = {}
    args['NUM_INVERSION'] = 250
    args['epoches'] = 50
    if dataset == 'cifar10':
        args['num_classes'] = 10
    elif dataset == 'mnist':
        args['num_classes'] = 10
    elif dataset == 'cifar100':
        args['num_classes'] = 100
    else:
        raise NotImplementedError
    return args

def improve_unlearning(model, device, train_loader, test_loader, wm_loader, arch_str, wm_str, dataset='cifar10', new_weight = 0.125,
                      inverse_num = 250, lr=0.01, fixed=False, re_batch_size=64, epoch=10, w2 = 1.0):
    print('##########', fixed, lr, re_batch_size, epoch, '##########')
    args = get_hyperparameters(dataset)
    NUM_INVERSION = inverse_num #args['NUM_INVERSION']
    assert NUM_INVERSION == 250

    num_classes = args['num_classes']
    
    # STAGE 1: Watermark Data Recovering
    store_path = f"reversed/reverse_lyf_{arch_str}_{wm_str+'nofix'}_{dataset}.t"
    print(store_path)
    if os.path.exists(store_path):
        class_xs_tensor = torch.load(store_path)
        print('found existing file at', store_path)
        print(f'loaded class_xs_tensor shape: {class_xs_tensor.shape}')
    else:
        raise RuntimeError('Class-wise inverted samples should be generated in advance.')
    
    # check class_xs_tensor.shape
    assert len(class_xs_tensor) == NUM_INVERSION * num_classes

    # STAGE 2: Watermark Unlearning
    if fixed:
        print('########## Assume the target watermark has fixed target label. ##########')
        print('Unlearning strategy: CE loss + random label')
        TARGET_LABEL = 6
        batch_xs = class_xs_tensor[TARGET_LABEL*NUM_INVERSION:(TARGET_LABEL+1)*NUM_INVERSION]
        proxy_wmk_indices, _ = split(batch_xs, model)
        batch_ys = torch.tensor([TARGET_LABEL] * NUM_INVERSION)
        rand_ys = torch.randint(low=TARGET_LABEL+1, high=TARGET_LABEL+num_classes, size=(NUM_INVERSION,)) % num_classes
        batch_ys[proxy_wmk_indices] = rand_ys[proxy_wmk_indices]
        # half maintain, half unlearn
        reverse_set = torch.utils.data.TensorDataset(batch_xs, batch_ys)
        reverse_loader = torch.utils.data.DataLoader(reverse_set, batch_size=re_batch_size, shuffle=True)
        if train_loader is None:
            print('Data-free setting.')
            finetune(model, reverse_loader, new_loader=reverse_loader, evaluate_loaders=[test_loader, wm_loader], epoch=epoch, lr=lr)
        else:
            print('With auxilary data.')
            finetune(model, train_loader, new_loader=reverse_loader, evaluate_loaders=[test_loader, wm_loader], epoch=epoch, lr=lr)
    else:
        print('########## Assume the target watermark has non-fixed target label. ##########')
        print('Unlearning strategy: CE loss + proxy label')
        LEAST_LIKELY_LABEL = 9 if dataset != 'cifar100' else 22
        SECOND_LEAST_LIKEY_LABEL = 5 if dataset != 'cifar100' else 33
        unlearn_xs = []
        unlearn_ys = []
        for i_class in range(num_classes):
            batch_xs = class_xs_tensor[i_class*NUM_INVERSION:(i_class+1)*NUM_INVERSION]
            proxy_wmk_indices, _ = split(batch_xs, model, dataset=='cifar100')
            unlearn_xs.append(batch_xs[proxy_wmk_indices])
            proxy_label = LEAST_LIKELY_LABEL if i_class != LEAST_LIKELY_LABEL else SECOND_LEAST_LIKEY_LABEL
            unlearn_ys.append(torch.tensor([proxy_label] * len(proxy_wmk_indices)))

        unlearn_xs_tensor = torch.concat(unlearn_xs)
        unlearn_ys_tensor = torch.concat(unlearn_ys)
        # half unlearn, drop other
        reverse_set = torch.utils.data.TensorDataset(unlearn_xs_tensor, unlearn_ys_tensor)
        reverse_loader = torch.utils.data.DataLoader(reverse_set, batch_size=re_batch_size, shuffle=True)
        
        if train_loader is None:
            raise RuntimeError('Data-free for non-fixed label watermark not supported.')
        print('With auxilary data.')
        

        origin_model = copy.deepcopy(model)
        origin_model.eval()

        ce_criterion = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        reverse_iter = iter(reverse_loader)

        for e in range(epoch):
            model.train()
            avg_loss1 = 0.0
            avg_loss2 = 0.0
            cnt = 0
            for inputs, targets in train_loader:
                try:
                    re_inputs, re_targets = next(reverse_iter)
                except StopIteration:
                    reverse_iter = iter(reverse_loader)
                    re_inputs, re_targets = next(reverse_iter)
                inputs, targets = inputs.to(device), targets.to(device)
                re_inputs, re_targets = re_inputs.to(device), re_targets.to(device)
                com_inputs = torch.concat((inputs, re_inputs))
                optim.zero_grad()
                com_outputs = model(com_inputs)
                len1 = len(inputs)
                w1 = 1.0
                w2 = w2
                loss1 = (w1 / (w1 + w2)) * ce_criterion(com_outputs[:len1], targets)
                loss2 = (w2 / (w1 + w2)) * ce_criterion(com_outputs[len1:], re_targets)
                loss = loss1 + loss2
                loss.backward()
                optim.step()
                avg_loss1 += loss1.data
                avg_loss2 += loss2.data
                cnt += 1
            avg_loss1 /= cnt
            avg_loss2 /= cnt
            print(e, avg_loss1.item(), avg_loss2.item())
            test(model, None, test_loader, device, type='Test')
            test(model, None, wm_loader, device, type='Watermark')

    return model

def split(batch_xs:torch.Tensor, model:torch.nn.Module):
    SPLIT_NUM = int(0.5 * len(batch_xs))
    # print(f"Split_num {SPLIT_NUM}")
    device = next(model.parameters()).device
    inputs = batch_xs.to(device)
    outputs = model(inputs, inspect=True)
    feats = outputs[-2].reshape(outputs[-2].shape[0], -1)

    feat_mean = torch.mean(feats, dim=0)
    feat_std = torch.std(feats, dim=0)
    neuron_impts = feat_mean / (feat_std + 1e-6)

    tmp_thre = torch.quantile(neuron_impts, 0.95)  # NOTE
    salient_neurons = torch.where(neuron_impts > tmp_thre)[0]
    sample_contribution = torch.mean(feats[:, salient_neurons], dim=1)
    sample_sorted_indices = sample_contribution.sort(descending=False)[1]
    proxy_wmk_indices = sample_sorted_indices[:SPLIT_NUM].detach().cpu()
    proxy_nor_indices = sample_sorted_indices[SPLIT_NUM:].detach().cpu()
    return proxy_wmk_indices, proxy_nor_indices
