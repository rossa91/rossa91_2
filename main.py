'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_path', default='./checkpoint', type=str, help='tracking bin change save_path')
parser.add_argument('--epoch', default=350, type=int, help='epoch')

#for quantization
parser.add_argument('--num_bits', default=4, type=int, help='quantization bit width')
parser.add_argument('--smooth_grad', default=None, type=float, help='smooth gradient how much')

#track train
parser.add_argument('--qtype', default=False, type=bool, help='tracking bin change mode train or not')

#for mixed quantization
parser.add_argument('--mixed', default=False, type=bool, help='mixed quantization or not')
parser.add_argument('--mask_load', default=None, type=str, help='To train mixed precision load the mask')


args = parser.parse_args()

print(args.smooth_grad)
#Set environment
if args.mixed is True :
  MASK = torch.load(args.mask_load)
else:
  MASK = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = QVGG('VGG19', num_bits=args.num_bits, mixed=args.mixed, mask=MASK, smooth_grad=args.smooth_grad)
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)


def track_train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    accum_check = dict([(k ,torch.zeros_like(v)) for (k, v) in net.named_parameters() if len(v.size())!= 1])

    save_path =  args.save_path + '/tracking'
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print('\nEpoch: {}  lr : {:f}' .format(epoch+1, optimizer.param_groups[0]['lr']))


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        with torch.no_grad():
          bwq = dict([(k ,quantize(v, num_bits=4, dequantize=False)) for (k, v) in net.named_parameters() if len(v.size())!= 1])

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
     
        with torch.no_grad():
          awq = dict([(k ,quantize(v, num_bits=4, dequantize=False)) for (k, v) in net.named_parameters() if len(v.size())!= 1])
          check = dict([(k, abs(v-awq[k])) for k,v in bwq.items()])
          accum_check = dict([(k, accum_check[k]+v) for k, v in  check.items()])

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    track_name = save_path+'/track{}.pth' .format(epoch+1) 
    print(track_name)
    torch.save(accum_check, track_name)

    print(accum_check['module.classifier.weight'])
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        
        if epoch <= 150:
          model_save_path = args.save_path + '/ckpt150.pth'
          torch.save(state, model_save_path)         
        elif epoch <= 250:
          model_save_path = args.save_path + '/ckpt250.pth'
          torch.save(state, model_save_path)
        else:
          model_save_path = args.save_path + '/ckpt350.pth'
          torch.save(state, model_save_path)

        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if args. qtype == True:
  print('==> track bin change mode')
  for epoch in range(start_epoch, start_epoch +args.epoch):
    track_train(epoch)
    test(epoch)
    scheduler.step()
else:
  print('==> normal training')
  for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
