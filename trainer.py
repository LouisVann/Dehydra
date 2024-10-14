'''Provides train and test function'''

import os
import logging
from tqdm import tqdm

import torch
import numpy as np

from helpers.utils import progress_bar

from helpers.pytorchtools import EarlyStopping

# train with train_loader for one epoch
def train(epoch, net, criterion, optimizer, train_loader, device,
          valid_loader=False, wmloader=False, tune_all=True):
    # print('\nEpoch: %d' % epoch)
    net.train()

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    train_acc = 0
    valid_acc = 0

    train_loss = 0
    correct = 0
    total = 0

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []  # list of batch, e.g. 32 32 32 4
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)  # each is a batch
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    
    
    # train_loader_p = tqdm(train_loader, desc='Train Epoch %d' % (epoch), disable=False)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print('\nBatch: %d' % batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = net(inputs)

        # calculate the loss
        loss = criterion(outputs, targets)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward(retain_graph=False)
        # perform a single optimization step (parameter update)
        optimizer.step()
        # record training loss
        train_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        train_acc = 100. * correct / total

    ######################
    # validate the model #
    ######################
    if valid_loader:
        correct = 0
        total = 0
        net.eval()  # prep model for evaluation
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = net(inputs)
                # calculate the loss
                loss = criterion(outputs, targets)
                # record validation loss
                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                valid_acc = 100. * correct / total

    # print training / validation statistics
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses) if valid_loader else 0.0

    logging.info(('Epoch %d: Train loss: %.3f | Valid loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (epoch, train_loss, valid_loss, valid_acc, correct, total)))

    return train_loss, valid_loss, train_acc, valid_acc


def test(net, criterion, loader, device, type='Test', return_loss=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    acc = 100. * correct / total
    test_loss /= (batch_idx + 1)
    if criterion is not None:
        logging.info('%s results: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (type, test_loss, acc, correct, total))
    else:
        logging.info('%s Acc: %.3f%% (%d/%d)'
                    % (type, acc, correct, total))
    return acc


def train_wo_wms(epochs, net, criterion, optimizer, scheduler, patience, train_loader, test_loader, valid_loader,
                 device, save_dir, save_model, history=dict()):
    logging.info("Training model without watermarks.")

    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    best_test_acc, best_epoch = 0, 0

    target_dir = save_dir
    assert os.path.isdir(target_dir)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=int(patience), verbose=True,
                                   path=os.path.join(target_dir, 'best.pth'),
                                   trace_func=logging.info)

    for epoch in range(epochs):
        logging.info('Learning rate: %.4f' % (scheduler.get_last_lr()[0]))
        train_loss, valid_loss, train_acc, valid_acc = train(epoch, net, criterion, optimizer, train_loader,
                                                                         device, valid_loader)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        test_acc = test(net, None, test_loader, device, type='Test')
        # logging.info("Test acc: %.3f%%" % test_acc)
        test_acc_list.append(test_acc)

        if avg_valid_losses[-1] < early_stopping.val_loss_min:  # bc this model will be saved
            best_test_acc = test_acc
            best_epoch = epoch

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_losses[-1], net)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        scheduler.step()

    torch.save(net.state_dict(), os.path.join(target_dir, 'final.pth'))

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['train_acc'] = train_acc
    history['valid_acc'] = valid_acc

    return best_test_acc, early_stopping.val_loss_min, best_epoch, history

# train with train_loader and wm_loader for one epoch
def train_on_augmented(epochs, device, net, optimizer, criterion, scheduler, patience, train_loader, test_loader,
                       valid_loader, wm_loader, save_dir, save_model, history):
    '''
    :param save_dir: target_dir
    :param save_model: args.runname updated with configs
    '''
    logging.info("Training on dataset augmented with trigger set.")

    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    wm_acc_list = []
    best_test_acc, best_wm_acc, best_epoch = 0, 0, 0
    
    target_dir = save_dir
    assert os.path.isdir(target_dir)

    early_stopping = EarlyStopping(patience=int(patience), verbose=True,
                                   path=os.path.join(target_dir, 'best.pth'),
                                   trace_func=logging.info)

    for epoch in range(epochs):
        logging.info('Learning rate: %.4f' % (scheduler.get_last_lr()[0]))
        train_loss, valid_loss, train_acc, valid_acc = train(epoch, net, criterion, optimizer, train_loader, 
                                                             device, valid_loader, wm_loader)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        test_acc = test(net, None, test_loader, device, type='Test')
        # logging.info("Test acc: %.3f%%" % test_acc)
        test_acc_list.append(test_acc)

        # logging.info("Testing triggerset (no train, test split).")
        wm_acc = test(net, None, wm_loader, device, type='Watermark')
        # logging.info("WM acc: %.3f%%" % wm_acc)
        wm_acc_list.append(wm_acc)

        if avg_valid_losses[-1] < early_stopping.val_loss_min:  # bc this model will be saved
            best_test_acc = test_acc
            best_wm_acc = wm_acc
            best_epoch = epoch

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_losses[-1], net)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        scheduler.step()

    torch.save(net.state_dict(), os.path.join(target_dir, 'final.pth'))

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['wm_acc'] = wm_acc_list
    history['train_acc'] = train_acc
    history['valid_acc'] = valid_acc

    # torch.save(torch.tensor([avg_train_losses, avg_valid_losses, test_acc_list, wm_acc_list]),
    #           os.path.join('results', save_model + '.pth'))

    return best_test_acc, best_wm_acc, early_stopping.val_loss_min, best_epoch, history

# for frontier_stitching only, use only wm_loader to train
def train_on_wms(epochs, device, net, optimizer, criterion, scheduler, wm_loader, test_loader, save_dir, save_model,
                 history):
    logging.info("Training model only on trigger set.")

    avg_train_losses = []
    avg_valid_losses = []
    test_acc_list = []
    wm_acc_list = []
    
    target_dir = save_dir
    assert os.path.isdir(target_dir)

    for epoch in range(epochs):
        logging.info('Learning rate: %.4f' % (scheduler.get_last_lr()[0]))
        # no valid_loader for wm_loader
        train_loss, valid_loss, train_acc, valid_acc = train(epoch, net, criterion, optimizer, wm_loader, device)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        test_acc = test(net, None, test_loader, device, type='Test')
        test_acc_list.append(test_acc)

        wm_acc = test(net, None, wm_loader, device, type='Watermark')
        wm_acc_list.append(wm_acc)
        
        scheduler.step()

    logging.info("Saving model.")
    torch.save(net.state_dict(), os.path.join(target_dir, 'best.pth'))

    history['train_losses'] = avg_train_losses
    history['valid_losses'] = avg_valid_losses
    history['test_acc'] = test_acc_list
    history['wm_acc'] = wm_acc_list
    history['train_acc'] = train_acc
    history['valid_acc'] = valid_acc

    # torch.save(torch.tensor([avg_train_losses, avg_valid_losses, test_acc_list, wm_acc_list]),
    #            os.path.join('results', save_model + '.pth'))

    return test_acc, wm_acc, None, epoch, history