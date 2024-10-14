import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer import test

def finetune(model_f, data_loader, new_loader=None, new_noise=0.0, evaluate_loaders=[None, None], lr=0.01, epoch=10):
    device = next(model_f.parameters()).device
    test_loader = evaluate_loaders[0]
    wm_loader = evaluate_loaders[1]
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model_f.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if new_loader is not None:
        new_iter = iter(new_loader)

    for e in range(epoch):
        model_f.train()
        avg_loss = 0.0
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
                targets = torch.concat((targets, targets_n))
            
            optim.zero_grad()
            outputs = model_f(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            avg_loss += loss.data
            cnt += 1
        avg_loss /= cnt
        print(e, avg_loss)
        test_acc = test(model_f, None, test_loader, device, type='Test')
        wm_acc = test(model_f, None, wm_loader, device, type='Watermark')


def want_imgs_from_class(want_class, max_num, dataloader):
    want_imgs = []
    data_iter = iter(dataloader)
    while len(want_imgs) < max_num:
        batch_xs, batch_ys = next(data_iter)
        for i in range(len(batch_xs)):
            x = batch_xs[i]
            y = batch_ys[i]
            if y != want_class:
                continue
            want_imgs.append(x)
            if len(want_imgs) >= max_num:
                break

    want_imgs = torch.stack(want_imgs)
    assert want_imgs.shape[0] == max_num
    return want_imgs


class ParameterMovement(object):
    def __init__(self, model_ori):
        params = []
        for _, param in model_ori.named_parameters():
            params.append(param.detach())
        self.origin_params = params
    
    def calc_movement(self, model_f):
        total_movement = 0.0
        for idx, (_, param) in enumerate(model_f.named_parameters()):
            total_movement += torch.norm(self.origin_params[idx] - param, p=2)
        return total_movement

def unlearn(model_f, model_ori, data_loader, new_loader, new_noise=0.0, evaluate_loaders=[None, None], beta=0.1):
    device = next(model_f.parameters()).device
    test_loader = evaluate_loaders[0]
    wm_loader = evaluate_loaders[1]
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model_f.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    new_iter = iter(new_loader)
    dist_regularizer = ParameterMovement(model_ori)

    for e in range(10):
        model_f.train()
        avg_loss = 0.0
        cnt = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            clean_batch_size = inputs.shape[0]
            try:
                inputs_n, _ = next(new_iter)
            except StopIteration:
                new_iter = iter(new_loader)
                inputs_n, _ = next(new_iter)
            inputs_n = inputs_n.to(device)
            unlearn_batch_size = inputs_n.shape[0]
            targets_n_not = torch.tensor([6] * unlearn_batch_size).to(device)
            if new_noise > 0.0:
                inputs_n += torch.randn_like(inputs_n) * new_noise
            inputs = torch.concat((inputs, inputs_n))            
            optim.zero_grad()
            outputs = model_f(inputs)

            # targets = torch.concat((targets, targets_n))
            loss = criterion(outputs[:clean_batch_size], targets) - criterion(outputs[clean_batch_size:], targets_n_not)
            loss += dist_regularizer.calc_movement(model_f) * beta
            loss.backward()
            optim.step()
            avg_loss += loss.data
            cnt += 1
        avg_loss /= cnt
        print(e, avg_loss)
        test_acc = test(model_f, None, test_loader, device, type='Test')
        wm_acc = test(model_f, None, wm_loader, device, type='Watermark')
    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class L2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.norm(x) / batch_size
    

class DeepInversionFeatureHook():
    '''
    https://github.com/NVlabs/DeepInversion/blob/master/deepinversion.py
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()
