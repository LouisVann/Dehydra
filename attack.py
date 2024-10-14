import logging
import os
import argparse

import torch
import models
import watermarks

from attacks.inversion_unlearning import inversion_removal
from attacks.improved_attack import improve_unlearning


# set up argument parser
from helpers.loaders import get_data_transforms, get_dataloader, get_dataset, get_wm_transform, get_wm_path
from helpers.utils import set_up_logger, get_trg_set, set_random_seed, check_data_ratio
from trainer import test

model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
watermarking_methods = sorted(
    watermark for watermark in watermarks.__dict__ if callable(watermarks.__dict__[watermark]))

parser = argparse.ArgumentParser(description='Perform attacks on models.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [default: cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar)')
parser.add_argument('--attack_type', default='nc', help='attack type. choices: pruning, fine-tuning')
parser.add_argument('--pruning_rates', nargs='+', default=[0.2], type=float, help='percentages (list) of how many weights to prune')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')

parser.add_argument('--batch_size', default=128, type=int, help='batch size for fine-tuning')
parser.add_argument('--wm_batch_size', default=32, type=int, help='batch size for fine-tuning')
parser.add_argument('--num_epochs', default=40, type=int, help='number of epochs for fine-tuning')
parser.add_argument('--patience', default=10, type=int, help='patience for transfer learning')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')

parser.add_argument('--method', default='ProtectingIP', type=str, choices=watermarking_methods,
                    help='watermarking method: ' + ' | '.join(watermarking_methods) + ' (default: weakness_into_strength)')
parser.add_argument('--trg_set_size', default=100, type=int, help='the size of the trigger set (default: 100)')
parser.add_argument('--wm_type', default='', type=str, help='e.g. content, noise, unrelated')
parser.add_argument('--loadmodel', default='lyf_resnet18_cifar10_ProtectingIPcontent_100', type=str, help='the model which the attack should be performed on')
parser.add_argument('--eps', default=0.1, type=float, help='eps for watermarking method')
parser.add_argument('--pattern_size', default=6, help='pattern size or num bits')
parser.add_argument('--save_model', action='store_true', help='save attacked model?')
parser.add_argument('--save_file', default="save_results_attacks.csv", help='file for saving results')

parser.add_argument('--tunealllayers', action='store_true', help='fine-tune all layers')
parser.add_argument('--reinitll', action='store_true', help='re initialize the last layer')

parser.add_argument('--data_ratio', default=0.01667, type=float, help='ratio of labeled dataset')
parser.add_argument('--OOD', action='store_true', help='use OOD dataset?')

parser.add_argument('--labeling_strategy', default=1, type=int, help='labeling_strategy in inversion_unlearning, for no-fixed watermark.')
parser.add_argument('--criterion_strategy', default=2, type=int, help='criterion_strategy in inversion_unlearning, for wm dataloader.')
parser.add_argument('--new_weight', default=0.125, type=float, help='weight of new_loss in inversion_unlearning.')
parser.add_argument('--w2', default=1.0, type=float)

parser.add_argument('--inverse_method', default="lyf", type=str,choices=["lyf", "bna"], help='Store path in iversion_unlearning')
parser.add_argument('--inverse_num', default=250, type=int, help='Inverse_num in inverse_removal')
parser.add_argument('--fixed', action='store_true', help='is fixed-label watermark?')
parser.add_argument('--free', action='store_true', help='data-free?')
parser.add_argument('--re_batch_size', default=64, type=int, help='batch size of reversed data')

parser.add_argument('--seed', default=0, type=int)

# gpu
parser.add_argument('--gpu', default='2', type=str, help='set gpu device (e.g. 0)')

args = parser.parse_args()
print(args)
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# set random seed for reproducibility
set_random_seed(seed=int(args.seed))
cwd = os.getcwd()
target_dir = os.path.join(cwd, 'checkpoint', args.loadmodel)

model = models.__dict__[args.arch](num_classes=args.num_classes)
model.load_state_dict(torch.load(os.path.join(target_dir, 'best.pth')))
model = model.to(device)

# set up log file
if args.attack_type == 'inversion' or args.attack_type == 'improved':
    info = str(args.loadmodel) + '_' + str(args.attack_type)
else:
    raise NotImplementedError('Attack is not implemented.')

logfile = os.path.join(target_dir, 'log_' + info + '.txt')
set_up_logger(logfile)
logging.info('\n\n')
logging.info('############################################################')
logging.info('Start attack. Attack type: ' + args.attack_type)
logging.info('Data ratio: ' + str(args.data_ratio))

check_data_ratio(args)

if args.dataset == 'cifar10':
    target_domain = 'cifar10'

    size_train = 50000  # or half
    size_test = 10000  # or half
    data_ratio = 0.01667
    # data_ratio = 0.5
elif args.dataset == 'mnist':
    target_domain = 'mnist'
    size_train = 60000  # or half
    size_test = 10000  # or half
    data_ratio = 0.0142857
elif args.dataset == 'cifar100':
    target_domain = 'cifar100'
    size_train = 50000  # or half
    size_test = 10000  # or half
    data_ratio = 0.01667

# prepare test loader
# print(f"cwd in attack {cwd}")
train_db_path = os.path.join(cwd, 'data')
test_db_path = os.path.join(cwd, 'data')
transform_train, transform_test = get_data_transforms(args.dataset)
train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, valid_size=args.data_ratio, testquot=None, size_train=size_train, size_test=size_test)
_, test_loader, valid_loader = get_dataloader(train_set, test_set, args.batch_size, valid_set, shuffle=False)#shuffle=True)


if args.OOD is False:
    print('clean samples', len(valid_set))
else:
    # transfer-based attack needs ood_loader
    if args.dataset == 'cifar10':
        proxy_dataset = 'cifar100'
        train_dataset_size = 50000  # cifar10
        test_dataset_size = 10000
        if args.attack_type == 'inversion' or args.attack_type == 'improved':
            proxy_num = 2000
            data_ratio = 0.03334
        else:
            raise NotImplementedError
    elif args.dataset == 'cifar100':
        proxy_dataset = 'cifar10'
        train_dataset_size = 50000  # cifar10
        test_dataset_size = 10000
        if args.attack_type == 'inversion' or args.attack_type == 'improved':
            proxy_num = 2000
            data_ratio = 0.03334
        else:
            raise NotImplementedError
    elif args.dataset == 'mnist':
        proxy_num = 2000
        proxy_dataset = 'emnist'
        train_dataset_size = 240000  # cifar10
        test_dataset_size = 40000
        data_ratio = 0.007143
    else:
        raise NotImplementedError
    
    transform_train_proxy, transform_test_proxy = get_data_transforms(proxy_dataset)

    proxy_train_set, proxy_test_set, proxy_valid_set = get_dataset(proxy_dataset, train_db_path, test_db_path, 
                                                                   transform_train_proxy, transform_test_proxy, 
                                                                   valid_size=data_ratio)
    tensor_X = []
    tensor_Y = []
    tmp_batch_size = 100
    model.eval()
    for i_batch in range(len(proxy_valid_set) // tmp_batch_size):
        tmp_xs = torch.stack([proxy_valid_set[i + i_batch*tmp_batch_size][0] for i in range(100)]).to(device)
        tmp_outs = model(tmp_xs)
        _, tmp_preds = torch.max(tmp_outs, dim=1)
        tensor_X.append(tmp_xs.detach().cpu())
        tensor_Y.append(tmp_preds.detach().cpu())
    tensor_X = torch.concat(tensor_X)
    tensor_Y = torch.concat(tensor_Y)
    print(tensor_X.shape, tensor_Y.shape)
    ood_set = torch.utils.data.TensorDataset(tensor_X, tensor_Y)
    print(f"OOD data scope {tensor_X.min()}/{tensor_X.max()}")
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=True)
    print('OOD samples', len(ood_set))

# prepare wm loader
wm_path = get_wm_path(args.method, args.dataset, wm_type=args.wm_type, model=args.arch, eps=args.eps, pattern_size=args.pattern_size)
transform = get_wm_transform(args.method, args.dataset)
trigger_set = get_trg_set(wm_path, 'labels.txt', args.trg_set_size, transform)

# print(f"Trigger_set scope {trigger_set[0][0].min()}/{trigger_set[0][0].max()}")

wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)

test_acc = test(model, None, test_loader, device, type='Clean Test')
wm_acc = test(model, None, wm_loader, device, type='Watermark')

if args.attack_type == 'inversion':
    logging.info('#'*20 + ' inversion ' + '#'*20)
    atk_train_loader = valid_loader if not args.OOD else ood_loader
    model = inversion_removal(model, device, atk_train_loader, test_loader, wm_loader, args.arch, args.method+args.wm_type, args.dataset, 
                              args.labeling_strategy, args.criterion_strategy, args.new_weight, args.inverse_method, args.inverse_num, args.lr)
elif args.attack_type == 'improved':
    logging.info('#'*20 + ' improved ' + '#'*20)
    atk_train_loader = valid_loader if not args.OOD else ood_loader
    if args.free:
        atk_train_loader = None
    model = improve_unlearning(model, device, atk_train_loader, test_loader, wm_loader, args.arch, args.method+args.wm_type, dataset=args.dataset,lr=args.lr, fixed=args.fixed, re_batch_size=args.re_batch_size, epoch=args.num_epochs, w2 = args.w2)
else:
    raise NotImplementedError('Attack is not implemented.')




