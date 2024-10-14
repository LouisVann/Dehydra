from helpers.loaders import get_data_transforms, get_dataset, get_dataloader
from helpers.pytorchtools import EarlyStopping
from trainer import train, test
from utils import DeepInversionFeatureHook, TVLoss, L2Loss

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        args['EPOCH_INVERSION'] = 800
        args['xs_shape'] = (args['NUM_INVERSION'], 3, 32, 32)
    else:
        raise NotImplementedError
    return args

def inversion_removal(model, device, train_loader, test_loader, wm_loader, arch_str, wm_str, dataset='cifar10', 
                      labeling_strategy_idx = 0, criterion_strategy_idx = 0, new_weight = 0.125, inverse_method = "lyf", 
                      inverse_num = 250, lr=0.01):
    args = get_hyperparameters(dataset)
    NUM_INVERSION = inverse_num #args['NUM_INVERSION']

    num_classes = args['num_classes']
    is_fixed = 'nofix'
    if is_fixed == 'fix':
        inverse_classes = [6]
    else:  # 'nofix'
        inverse_classes = list(range(num_classes)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # STAGE 1: Watermark Data Recovering

    if inverse_method == "lyf":
        store_path = f"reversed/reverse_lyf_{arch_str}_{wm_str+is_fixed}_{dataset}.t"
        
        if os.path.exists(store_path):
            class_xs_tensor = torch.load(store_path)
            print('found existing file at', store_path)
            print(f'loaded class_xs_tensor shape: {class_xs_tensor.shape}')
        else:
            class_xs_tensor = inverse_class(model, device, args, inverse_classes)
            torch.save(class_xs_tensor.detach().cpu(), store_path)
            print('reversed samples are saved to', store_path)
    elif inverse_method == "bna":
        # f"reversed/reverse_bna_nodiv_{method_path_name}_{dataset}.t")
        store_path = f"reversed/reverse_bna_{arch_str}_{wm_str}_{dataset}.t"
        if os.path.exists(store_path):
            class_xs_tensor = torch.load(store_path)
            print('found existing file at', store_path)
            
            print(f'loaded class_xs_tensor shape: {class_xs_tensor.shape}')
        else:
            class_xs_tensor = inverse_class(model, device, args, inverse_classes)
            torch.save(class_xs_tensor.detach().cpu(), store_path)
            print('reversed samples are saved to', store_path)
    else:
        raise ValueError(f"Inverse_method {inverse_method} invalid, only [lyf, bna]")

    print(f"class_xs_tensor max{class_xs_tensor.max()}, min {class_xs_tensor.min()}")
    
    # check class_xs_tensor.shape
    if class_xs_tensor.shape[0] != NUM_INVERSION * num_classes:
        class_xs_list = []
        for idx in range(num_classes):
            class_xs_list.append(class_xs_tensor[args['NUM_INVERSION']*idx: args['NUM_INVERSION']*idx + NUM_INVERSION])
        class_xs_tensor = torch.concat(class_xs_list)
        print(f"inverse class_xs_tensor shape: {class_xs_tensor.shape}")

    # STAGE 2: Watermark Unlearning
    new_labeling_list = ['proxy', 'soft', 'few-shot', 'few-shot_avg']
    new_labeling_idx = labeling_strategy_idx
    new_labeling = new_labeling_list[new_labeling_idx]
    assert new_labeling_idx == 1

    new_criterion_list = ['CEloss', 'KLloss']
    new_criterion_idx = criterion_strategy_idx
    new_criterion = new_criterion_list[new_criterion_idx]
    assert new_criterion_idx == 1

    logging.info(f"Trying new labeling ({new_labeling}), loss({new_criterion}) for no fixed watermrak remove.")
    
    class_ys = []
    if is_fixed == 'fix':
        assert len(inverse_classes) == 1
        class_ys.append(torch.randint(low=inverse_classes[0]+1, high=inverse_classes[0]+num_classes, size=(NUM_INVERSION,)) % num_classes)
    else:
        if new_labeling == 'soft':
            class_ys =  [torch.stack([torch.tensor([float(1/num_classes)] * num_classes)] * NUM_INVERSION)] * num_classes
        else:
            raise ValueError(f"Not implemented yet.")
    class_ys_tensor = torch.concat(class_ys)
    print(class_ys_tensor.shape)

    total_reverse_set = torch.utils.data.TensorDataset(class_xs_tensor, class_ys_tensor)
    total_reverse_loader = torch.utils.data.DataLoader(total_reverse_set, batch_size=128, shuffle=True)

    print('finetuning with clean samples and inversed samples, without noise.')
    inverse_finetune(model, train_loader, total_reverse_loader, new_noise=0.0, evaluate_loaders=[test_loader, wm_loader] ,lr=lr, epoch=args['epoches'], labeling_strategy = new_labeling, criterion_strategy = new_criterion, num_classes = num_classes, new_weight = new_weight)  # change noise
    return model

def inverse_finetune(model_f, data_loader, new_loader=None, new_noise=0.0, evaluate_loaders=[None, None], lr=0.01, epoch=10, labeling_strategy = None, criterion_strategy = None, num_classes = 10, new_weight = 0.125):
    device = next(model_f.parameters()).device
    test_loader = evaluate_loaders[0]
    wm_loader = evaluate_loaders[1]

    clean_criterion = nn.CrossEntropyLoss()

    if criterion_strategy == 'CEloss':
        new_criterion = nn.CrossEntropyLoss()
    elif criterion_strategy == 'KLloss':
        new_criterion = nn.KLDivLoss(reduce='batchmean')
    else:
        raise ValueError(f"new_criterion Values {new_criterion} not in list.")
        
    optim = torch.optim.SGD(model_f.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    print(lr, new_weight)

    if new_loader is not None:
        new_iter = iter(new_loader)

    for e in range(epoch):  
        model_f.train()
        avg_loss = 0.0
        avg_clean_loss = 0.0
        avg_new_loss = 0.0
        cnt = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if new_loader is not None:
                try:
                    inputs_n, targets_n = next(new_iter)
                except StopIteration:
                    new_iter = iter(new_loader)
                    inputs_n, targets_n = next(new_iter)
                inputs_n, targets_n = inputs_n.to(device), targets_n.to(device)
                if new_noise > 0.0:
                    inputs_n += torch.randn_like(inputs_n) * new_noise
                inputs = torch.concat((inputs, inputs_n))

            optim.zero_grad()
            outputs = model_f(inputs)

            clean_outputs = outputs[:len(inputs) - len(inputs_n), :]
            new_outputs = outputs[len(inputs) - len(inputs_n):, :]

            clean_loss = clean_criterion(clean_outputs, targets)

            # For new loss
            if criterion_strategy == 'CEloss':
                # hard_label
                new_loss = new_criterion(new_outputs, targets_n)
            elif criterion_strategy == 'KLloss':
                new_outputs = F.log_softmax(new_outputs, dim = 1)
                assert labeling_strategy == 'soft'
                new_loss = new_criterion(new_outputs, targets_n)
            
            loss = clean_loss + new_weight * new_loss  
            loss.backward()
            optim.step()
            avg_loss += loss.data
            avg_clean_loss += clean_loss.data
            avg_new_loss += new_loss.data
            cnt += 1
        avg_loss /= cnt
        avg_clean_loss /= cnt
        avg_new_loss /= cnt
        print('%d %.4f=%.4f+%.4f*%.2f' % (e, avg_loss.item(), avg_clean_loss.item(), avg_new_loss.item(), new_weight))
        test_acc = test(model_f, None, test_loader, device, type='Test')
        wm_acc = test(model_f, None, wm_loader, device, type='Watermark')


def inverse_class(model, device, args, inverse_classes):
    model.eval()
    NUM_INVERSION = args['NUM_INVERSION']
    mean = torch.tensor(args['mean']).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device)
    std = torch.tensor(args['std']).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device)
    EPOCH_INVERSION = args['EPOCH_INVERSION']
    xs_shape = args['xs_shape']
    
    loss_r_feature_layers = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    bn_reg_weight = 0.1
    if args['dataset'] == 'cifar100':
        bn_reg_weight = 0.01

    tv_criterion = TVLoss()
    l2_criterion = L2Loss()
    criterion = nn.CrossEntropyLoss()
    class_xs = []
    for i_class in inverse_classes:
        xs_v = torch.randn(size=xs_shape).to(device)
        xs_v.requires_grad = True
        targets = torch.tensor([i_class] * NUM_INVERSION).to(device)
        optimizer = torch.optim.Adam([xs_v], lr=0.1, betas=[0.5, 0.9], eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_INVERSION, eta_min=1e-4)

        for i in range(EPOCH_INVERSION):
            optimizer.zero_grad()
            imgs = (torch.tanh(xs_v) + 1.0) / 2.0
            imgs = (imgs - mean) / std
            
            outs = model(imgs)
            cls_loss = criterion(outs, targets)
            l2_reg = l2_criterion(imgs) * 0.01
            tv_reg = tv_criterion(imgs) * 0.03

            if len(loss_r_feature_layers) > 0:
                rescale = [1.] + [1. for _ in range(len(loss_r_feature_layers)-3)] + [0., 0.]
                print(f"bn_reg {bn_reg_weight}")
            else:
                bn_reg = torch.tensor(0.0)

            loss = cls_loss + l2_reg + tv_reg + bn_reg
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 100 == 99:
                logging.info('Class %d\t Epoch %d\t Losses %.4f %.4f %.4f %.4f' % \
                            (i_class, i, cls_loss.item(), l2_reg.item(), tv_reg.item(), bn_reg.item()))


        with torch.no_grad():
            imgs = (torch.tanh(xs_v) + 1.0) / 2.0
            imgs = (imgs - mean) / std
            outs = model(imgs)
            _, predicted = torch.max(outs.data, dim=1)
            atk_success = predicted.eq(targets).sum().item()
        print('Class %d success num: %d' % (i_class, atk_success))
        class_xs.append(imgs.detach().cpu())

        del xs_v, targets, outs, cls_loss, l2_reg, tv_reg, loss
        torch.cuda.empty_cache()
    
    class_xs_tensor = torch.concat(class_xs)
    print(class_xs_tensor.shape)        
    return class_xs_tensor
