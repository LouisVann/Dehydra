"""Adversarial Frontier Stitching (Merrer et al., 2019)

- needs pretrained model
- generating adversarial inputs with fast gradient sign method

Implementation based on: https://github.com/dunky11/adversarial-frontier-stitching/ (original in tensorflow)"""
from helpers.loaders import get_wm_transform
from watermarks.base import WmMethod

import os
import logging

import torch

from helpers.utils import find_tolerance, fast_gradient_sign, save_triggerset, get_trg_set
from trainer import train_on_wms, train_on_augmented


class FrontierStitching(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'frontier_stitching')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

    def gen_watermarks(self, model, criterion, device, train_loader, eps):#, seed = 0):
        true_advs = list()
        false_advs = list()

        # clean_true = list()
        # clean_false = list()

        max_true_advs = max_false_advs = self.size // 2

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # generate adversary
            x_advs = fast_gradient_sign(x, y, model, criterion, device, eps)

            y_preds = torch.argmax(model(x), dim=1)
            y_pred_advs = torch.argmax(model(x_advs), dim=1)

            for idx, (x_adv, y_pred_adv, y_pred, y_true) in enumerate(zip(x_advs, y_pred_advs, y_preds, y)):
                # x_adv is a true adversary
                if y_pred == y_true and y_pred_adv != y_true and len(true_advs) < max_true_advs:
                    true_advs.append((x_adv, y_true))
                    print('true adv', len(true_advs))

                    # clean_true.append((x[idx],y_true))

                # x_adv is a false adversary
                if y_pred == y_true and y_pred_adv == y_true and len(false_advs) < max_false_advs:
                    false_advs.append((x_adv, y_true))
                    print('false adv', len(false_advs))

                    # clean_false.append((x[idx],y_true))

                if len(true_advs) == max_true_advs and len(false_advs) == max_false_advs:
                    break

        if self.save_wm:
            save_triggerset(true_advs + false_advs, os.path.join(self.path, self.arch, str(self.eps)), self.dataset)#, str(seed))

            # save_triggerset(clean_true + clean_false, os.path.join(os.path.join(os.getcwd(), 'data', 'origin_pic', 'frontier_stitching'), self.arch, str(self.eps)), self.dataset)

        self.trigger_set = true_advs + false_advs
        return self.trigger_set

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader, device, save_dir):

        logging.info("Embedding watermarks.")

        # load model
        # net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel, 'best.pth')))

        # generating watermarks
        logging.info("Generating watermarks.")
        # self.gen_watermarks(net, criterion, device, train_loader, self.eps)
        transform = get_wm_transform('FrontierStitching', self.dataset)

        self.trigger_set = get_trg_set(os.path.join(self.path, self.arch, str(self.eps), self.dataset), 'labels.txt',
                                       self.size, transform=transform)

        self.loader()

        if self.embed_type == 'only_wm':
            real_acc, wm_acc, val_loss, epoch, self.history = train_on_wms(self.epochs_w_wm, device, net, optimizer, criterion, scheduler, self.wm_loader,
                                  test_loader, save_dir, None, self.history)

        elif self.embed_type == 'augmented':
            real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion, scheduler, self.patience,
                                                  train_loader, test_loader, valid_loader, self.wm_loader, save_dir,
                                                  None, self.history)


        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch
