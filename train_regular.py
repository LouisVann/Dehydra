"""Training models without watermark."""

import argparse
import traceback

from babel.numbers import format_decimal

from torch.backends import cudnn

import models

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *

from trainer import train_wo_wms

# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
# print('models : ', model_names)


# set up argument parser
parser = argparse.ArgumentParser(description='Train models without watermarks.')

# model and dataset
parser.add_argument('--dataset', default='mnist', help='the dataset to train on [cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='simplenet_mnist', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar10)')

# hyperparameters
parser.add_argument('--runname', default='cifar10_custom_cnn', help='the exp name')
parser.add_argument('--epochs_wo_wm', default=60, type=int, help='number of epochs trained without watermarks')
parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--patience', default=20, help='early stopping patience (default 20)')

# cuda
parser.add_argument('--gpu', default='0', type=str, help='set gpu device (e.g. 0)')

# for testing with a smaller subset
parser.add_argument('--test_quot', default=None, type=int,
                    help='the quotient of data subset (for testing reasons; default: None)')

# experiments
parser.add_argument('--save_file', default="save_results.csv", help='file for saving results')

# random seed
parser.add_argument('--seed', default=0, help='the random seed for init.')

args = parser.parse_args()
# print(args)
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

set_random_seed(seed=int(args.seed))

# random init
batch_size_set = [32, 64, 128]
lr_set = [0.1, 0.05, 0.01, 0.001]
epoch_set = [60, 80, 100]
if args.dataset == 'mnist' and args.arch == 'lenet5':
    epoch_set = [20, 22, 24, 26]
    lr_set = [0.01, 0.001]
    args.patience = 5

print(args)
try:
    cwd = os.getcwd()
    null_dir = os.path.join(cwd, 'null_models')
    os.makedirs(null_dir, exist_ok=True)

    arch_name = '_'.join((args.dataset, args.arch))
    arch_dir = os.path.join(null_dir, arch_name)
    os.makedirs(arch_dir, exist_ok=True)


    target_dir = os.path.join(arch_dir, f"{get_max_index(arch_dir, suffix='null_model').zfill(5)}_null_model")
    os.makedirs(target_dir, exist_ok=True)
    
    configfile = os.path.join(target_dir, 'configs.txt')  # time.strftime("%Y%m%d-%H%M%S_") + 
    logfile = os.path.join(target_dir, 'logs.txt')
    set_up_logger(logfile)

    # save configuration parameters
    with open(configfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    # set up paths for dataset
    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')

    # load train, valid and test set
    valid_size = 0.1
    transform_train, transform_test = get_data_transforms(args.dataset)
    train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test,
                                                 valid_size, testquot=args.test_quot)
    train_loader, test_loader, valid_loader = get_dataloader(train_set, test_set, args.batch_size, valid_set, shuffle=True)

    # set up loss
    criterion = nn.CrossEntropyLoss()

except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)

try:
    # create new model
    logging.info('Building model. new Model: ' + args.arch)
    net = models.__dict__[args.arch](num_classes=args.num_classes)
    net.to(device)

    # set up optimizer and scheduler
    optimizer, scheduler = set_up_optim_sched(net, args.lr, args.optim, args.sched, lradj=args.lradj, T_max=args.epochs_wo_wm)

    logging.info('Training model.')

    real_acc, val_loss, epoch, history = train_wo_wms(args.epochs_wo_wm, net, criterion, optimizer, scheduler,
                                                      args.patience, train_loader, test_loader, valid_loader,
                                                      device, target_dir, args.runname)

    # save results to csv
    csv_args = [getattr(args, arg) for arg in vars(args)] + [format_decimal(real_acc.item(), locale='en_US'),
                                                             None, None, None, None, val_loss, epoch]

    save_results(csv_args, os.path.join(cwd, args.save_file))

    del net
    del optimizer
    del scheduler

except Exception as e:
    msg = 'An error occurred during training in ' + args.runname + ': ' + str(e)
    logging.error(msg)

    traceback.print_tb(e.__traceback__)