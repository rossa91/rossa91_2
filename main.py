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
from auxil import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--num_bits', default=4, type=int, help='quantization_bits')
parser.add_argument('--load_path', default=None, type=str, help='load_path')
parser.add_argument('--save_path', default='./checkpoint', type=str, help='save_path')
parser.add_argument('--qtype', default=False, type=bool, help='Quantization Type or Not')
parser.add_argument('--epoch', default=350, type=int, help='Epoch')


args = parser.parse_args()




device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
last_epoch = start_epoch + args.epoch


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


# net = VGG('VGG9')
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
net = QVGG('VGG9', num_bits=args.num_bits, mixed=False, mask=None)
# net = QMobileNet(num_bits=args.num_bits)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('---- checkpoint path')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

# re-training
if args.load_path is not None:
  print('==> load from checkpoint...')
  print('==> for retraining model..')
  if device == 'cuda':
    load_ckpt = torch.load(args.load_path)
  else :
    load_ckpt = torch.load(args.load_path, map_location=torch.device('cpu'))
    ckpt2 = {}
    for k, v in load_ckpt['net'].items():
      if 'module' in k:
        ckpt2[k[len('module.'):]] = v
    load_ckpt = ckpt2

  net.load_state_dict(load_ckpt, False)  

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)


#Training for tracking quantized bin change
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

        bwq = dict([(k ,quantize(v, num_bits=4, dequantize=False)) for (k, v) in net.named_parameters() if len(v.size())!= 1])

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
     
        awq = dict([(k ,quantize(v, num_bits=4, dequantize=False)) for (k, v) in net.named_parameters() if len(v.size())!= 1])

        check = dict([(k, abs(v-awq[k])) for k,v in bwq.items()])
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accum_check = dict([(k, accum_check[k]+v) for k, v in  check.items()])


    track_name = save_path+'/track{}.pth' .format(epoch+1) 
    print(track_name)
    torch.save(accum_check, track_name)

    print(accum_check['module.classifier.weight'])
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# Training for quantized model
def qtrain(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    


    print('\nEpoch: {}  lr : {}' .format(epoch, optimizer.param_groups[0]['lr']))
    print('Saving for tracking....')
    save_path =  args.save_path + '/tracking'
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)



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
        

        save_name = save_path+'/iter{}.pth' .format((epoch+1)*batch_idx) 
        torch.save(net.state_dict, save_name)
   
    


# Training for normal 
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    print('\nEpoch: {}  lr : {:f}' .format(epoch+1, optimizer.param_groups[0]['lr']))


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

    # Save best accuracy model
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    state_fg_path = 'epoch{}.pth' .format(epoch)
    state_for_graph = { 'acc' : acc, 'loss' : loss }
    torch.save(state_for_graph, './checkpoint/'+state_fg_path)




if args.qtype == True:
  print('qtrain mode')
else:
  print('normal train mode')

for epoch in range(start_epoch, last_epoch):
  if args.qtype == True:
    track_train(epoch)
  else:
    train(epoch)
  
  scheduler.step()
  test(epoch)

if args.qtype == True :
#  accum_all_track(last_epoch, load_path='./checkpoint/tracking')
   accum_all_track(load_path='./checkpoint/tracking')
   make_mask(load_path='./checkpoint/tracking', mixed_portion=0.2)



