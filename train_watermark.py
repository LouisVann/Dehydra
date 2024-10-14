"""implementing watermarking methods for"""

import argparse
import time
import traceback

from babel.numbers import format_decimal

from torch.backends import cudnn

import models
import watermarks

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *
from trainer import test


# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
# print('models : ', model_names)
# lenet1, resnet18, cnn_cifar10

# possible watermarking methods to use
watermarking_methods = sorted(
    watermark for watermark in watermarks.__dict__ if callable(watermarks.__dict__[watermark]))
# print('watermarks: ', watermarking_methods)

# set up argument parser
parser = argparse.ArgumentParser(description='Train models with watermarks.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar)')

# watermark related
parser.add_argument('--method', default='ProtectingIP', type=str, choices=watermarking_methods,
                    help='watermarking method: ' + ' | '.join(watermarking_methods) + ' (default: ProtectingIP)')
parser.add_argument('--wm_type', default=None, type=str, help='wm type for ProtectingIP: content, unrelated, noise')
parser.add_argument('--save_wm', action='store_true', help='save generated watermarks?')
parser.add_argument('--runname', default='tmp', type=str, help='the exp name')
parser.add_argument('--trg_set_size', default=100, type=int, help='the size of the trigger set (default: 100)')
parser.add_argument('--thresh', default=0.05, type=float, help='threshold for watermark verification')
parser.add_argument('--embed_type', default='', choices=['', 'fromscratch', 'pretrained', 'only_wm', 'augmented'],
                    help='either fromscratch or pretrained or only_wm or augmented')

parser.add_argument('--loadmodel', default='', help='path which model should be load for pretrained embed type')
parser.add_argument('--eps', default=0.25, help='epsilon for FrontierStitching or WMEmbeddedSystems')
parser.add_argument('--lmbda', default=100, help='lambda for PiracyResistant')
parser.add_argument('--pattern_size', default=6, help='patternsize for PiracyResistant and WMEMbeddedSystems')

# hyperparameters
parser.add_argument('--epochs_w_wm', default=60, type=int, help='number of epochs trained with watermarks')
# parser.add_argument('--epochs_w_wm', default=1, type=int, help='number of epochs trained with watermarks')
parser.add_argument('--epochs_wo_wm', default=0, type=int, help='number of epochs trained without watermarks')
parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=32, type=int, help='the wm batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', type=str, help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', type=str, help='scheduler (default MultiStepLR)')
parser.add_argument('--patience', default=10, type=int, help='early stopping patience (default 10)')
parser.add_argument('--temp', default=1000.0, type=float, help='temperature for EWE watermark')
parser.add_argument('--seed', default=0, type=int)

# gpu
parser.add_argument('--gpu', default='0', type=str, help='set gpu device (e.g. 0)')

# for testing with a smaller subset
parser.add_argument('--test_quot', default=None, type=int,
                    help='the quotient of data subset (for testing reasons; default: None)')

# experiments
parser.add_argument('--save_file', default="save_results.csv", help='file for saving results')


args = parser.parse_args()
print(args)
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# set random seed for reproducibility
set_random_seed(seed=int(args.seed))
    
try:
    cwd = os.getcwd()
    # create save_dir, results_dir and loss_plots_dir
    ckpt_dir = os.path.join(cwd, 'checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    watermark_name = args.method
    if args.wm_type is not None:
        watermark_name += args.wm_type
    args.runname = '_'.join((args.runname, args.arch, args.dataset, watermark_name, str(args.trg_set_size)))
    # args.runname = runname + "_" + args.optim + "_" + args.sched + "_" + str(args.trg_set_size)
    
    target_dir = os.path.join(ckpt_dir, args.runname)
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
    valid_size = 0.1  # https://arxiv.org/abs/1512.03385 uses 0.1 for resnet (0.1 split from train_set)
    transform_train, transform_test = get_data_transforms(args.dataset)
    train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test,
                                                 valid_size, testquot=args.test_quot)
    train_loader, test_loader, valid_loader = get_dataloader(train_set, test_set, args.batch_size, valid_set, shuffle=True)

    criterion = nn.CrossEntropyLoss()

except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)

try:
    generation_time = 0
    embedding_time = 0
    real_acc, wm_acc, success, false_preds, theta = None, None, None, None, None

    # create new model
    logging.info('Building model. new Model: ' + args.arch)
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    model = model.to(device)

    # set up optimizer and scheduler
    optimizer, scheduler = wm_set_up_optim_sched(model, args.lr, args.optim, args.sched, lradj=args.lradj, T_max=args.epochs_w_wm+args.epochs_wo_wm, method = args.method)
    logging.info('Training model with watermarks. Method: ' + args.method)
    logging.info('Trigger set size: ' + str(args.trg_set_size))
    # initialize method
    wm_method = watermarks.__dict__[args.method](args)

    # embed watermark
    start_time = time.time()
    real_acc, wm_acc, val_loss, epoch = wm_method.embed(model, criterion, optimizer, scheduler, train_set, test_set,
                                                        train_loader, test_loader, valid_loader, device, target_dir)  # target_dir is called 'save_dir' in watermarks
    embedding_time = time.time() - start_time
    logging.info("Time for embedding watermarks: %s" % embedding_time)

    # verify watermark, using 'best.pth' instead of 'final.pth'
    success, false_preds, theta = wm_method.verify(model, device)  # 'best.pth' parameters loaded
    
    test_acc = test(model, None, test_loader, device, type='Best test')

    # save results to csv
    csv_args = [getattr(args, arg) for arg in vars(args)] + [format_decimal(real_acc.item(), locale='en_US'),
                                                                format_decimal(wm_acc.item(), locale='en_US'),
                                                                bool(success.item()), false_preds.item(), theta,
                                                                val_loss, epoch+1, generation_time, embedding_time]
    save_results(csv_args, os.path.join(cwd, args.save_file))
    
    with open(os.path.join(target_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(wm_method.history, f, pickle.HIGHEST_PROTOCOL)

    del model
    del optimizer
    del scheduler
    del wm_method

except Exception as e:
    msg = 'An error occurred during training in ' + args.runname + ': ' + str(e)
    logging.error(msg)

    traceback.print_tb(e.__traceback__)