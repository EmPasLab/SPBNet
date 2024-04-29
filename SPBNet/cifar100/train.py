from net import xresnet18
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms as T
from tqdm import tqdm
from logger import get_logger
from torchsummary import summary
from utils import *

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
parser.add_argument('--data', default='./data/cifar100', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=40, type=int, metavar='N', help='number of data loading workers (default: 40)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--outputfile', action='store', default='./log/result.out', help='output file')

parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

CLASSES = 100

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

    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)

    print("###################################################################")
    print('w_bits: ', kwarg.get("w_bits"))
    print('a_bits: ', kwarg.get("a_bits"))
################################################################################

    transform_train = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         transforms.RandomErasing(),
     ])
 
    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])

    train_set = CIFAR100(root=args.data, train=True, transform=transform_train, download=False)
    val_set = CIFAR100(root=args.data, train=False, transform=transform_test, download=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 실험 조건
    model = xresnet18(**kwarg)
    model = nn.DataParallel(model).cuda()


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()


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
            momentum=0.9, lr=args.learning_rate)
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
    elif args.scheduler == 'foursteps':
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 200, 260, 320], gamma=0.2)

    logging.info(f'W_bits: {kwarg.get("w_bits")} | A_bits: {kwarg.get("a_bits")}')

    start_epoch = 0
    best_top1_acc= 0

    # initialize the model
    if not args.pretrained:
        print('==> not pretrained...')
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained, map_location='cuda:0')
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
        elif args.scheduler == 'foursteps':
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 200, 260, 320], gamma=0.2)

    if args.evaluate:
      valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
      exit(0)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer, scheduler)
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

        #if args.noscheduler == False:
        scheduler.step()
        epoch += 1

    #training_time = (time.time() - start_t) / 36000
    training_time = (time.time() - start_t) / 3600 # unit: one second
    print('total training time = {} hours'.format(training_time))
#}}}

def train(epoch, train_loader, model, criterion, optimizer, scheduler):
#{{{
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
    #adjust_learning_rate(optimizer,epoch)
    #scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

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
#}}}

def validate(epoch, val_loader, model, criterion, args):
#{{{
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
#}}}


if __name__ == '__main__':
    kwarg = {'w_bits': args.w_bits, 'a_bits': args.a_bits}
    main(**kwarg)
