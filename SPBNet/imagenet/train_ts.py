from xmobilenet import xreactnet
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms as T
from tqdm import tqdm
from logger import get_logger
from torchsummary import summary
from utils import *

import torch.nn as nn
from KD_loss import *
import torchvision.models as models
import logging
import numpy as np
import time

################################################################################
parser = argparse.ArgumentParser("ResNet")
parser.add_argument('--w_bits', type=int, default= 4, help='bits for weights')
parser.add_argument('--a_bits', type=int, default= 4, help='bits for activations')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--scheduler', action='store', default='multistep_200', help='scheduler')
parser.add_argument('--optimizer', action='store', default='adam', help='optimizer mode')
parser.add_argument('--pretrained', action='store', default=None, help='the path to the pretrained model')

parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--data', default='./data/imagenet', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=40, type=int, metavar='N', help='number of data loading workers (default: 40)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--outputfile', action='store', default='./log/result.out', help='output file')
parser.add_argument('--trap', action='store', default='trap', help='output file')

parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--board', type=str, default='./results', help='path for saving tensorboard')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model') 

parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

CLASSES = 1000

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
################################################################################

def main(**kwarg):
################################################################################
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True # random
    logging.info("args = %s", args)

    print("###################################################################")
    print('w_bits: ', kwarg.get("w_bits"))
    print('a_bits: ', kwarg.get("a_bits"))
################################################################################

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_dataset = torchvision.datasets.ImageNet(
        args.data, split='train', 
        transform=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))

    val_dataset = torchvision.datasets.ImageNet(
        args.data, split='val', 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    model = xreactnet(**kwarg)
    model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    criterion_kd = DistributionLoss()


    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    if args.optimizer== 'adam': 
      print('optimizer: adam')
      optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)
    elif args.optimizer == 'adamw': 
      print('optimizer: adamw')
      optimizer = torch.optim.AdamW(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)
    elif args.optimizer == 'sgd': 
      print('optimizer: sgd')
      optimizer = torch.optim.SGD(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate, momentum=0.9)
    else: 
      print('There is no optimizer setting; so exit!!')
      sys.exit(1)

    # original code
    if args.scheduler == 'multistep_200':
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1)
    elif args.scheduler == 'multistep_100':
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.1)
    elif args.scheduler == 'lambda':
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    elif args.scheduler == 'exponent':
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    elif args.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0, last_epoch=-1, verbose=False)
    elif args.scheduler == 'constant':
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=1)
    elif args.scheduler == 'twostages':
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 320], gamma=0.2)

    logging.info(f'W_bits: {kwarg.get("w_bits")} | A_bits: {kwarg.get("a_bits")}')

    start_epoch = 0
    best_top1_acc= 0

    # initialize the model
    if not args.pretrained:
        print('==> not pretrained...')
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained, map_location='cuda:0')
        #best_top1_acc = pretrained_model['best_top1_acc']
        model.load_state_dict(pretrained_model['state_dict']) 

    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
        if args.scheduler == 'multistep_200':
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1, last_epoch=start_epoch)
        elif args.scheduler == 'multistep_100':
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.1, last_epoch=start_epoch)
        elif args.scheduler == 'lambda':
          scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=start_epoch)
        elif args.scheduler == 'exponent':
          scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=start_epoch)
        elif args.scheduler == 'cosine':
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0, last_epoch=start_epoch, verbose=False)
        elif args.scheduler == 'constant':
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=1, last_epoch=start_epoch)
        elif args.scheduler == 'twostages':
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 320], gamma=0.2, last_epoch=start_epoch)     

    if args.evaluate:
      valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
      exit(0)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:

        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, model_teacher, criterion_kd, optimizer)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        outputfile_handler =open(args.outputfile, 'a+')

        print('epoch {epoch} train_loss {train_loss:.3f} train_acc@1 {train_top1_acc:.3f} train_acc@5 {train_top5_acc:.3f} val_loss {val_loss:.3f} val_acc@1 {val_top1_acc:.3f} val_acc@5 {val_top5_acc:.3f}'
              .format(epoch=epoch, train_loss=train_obj, train_top1_acc=train_top1_acc, train_top5_acc=train_top5_acc, val_loss=valid_obj, val_top1_acc=valid_top1_acc, val_top5_acc=valid_top5_acc), file=outputfile_handler)
        outputfile_handler.close()

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)

        scheduler.step()
        epoch += 1

    training_time = (time.time() - start_t) / 3600 # unit: one second
    print('total training time = {} hours'.format(training_time))
#}}}


def train(epoch, train_loader, model, model_teacher, criterion, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        logits_teacher = model_teacher(images)
        loss = criterion(logits, logits_teacher)
        #loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    kwarg = {'w_bits': args.w_bits, 'a_bits': args.a_bits}
    main(**kwarg)
