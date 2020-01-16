from __future__ import print_function
import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision

from dataset import Dataset

from models.BiSeNet import BiSeNet
from models.mobilenetv2_seg_small import MobileNetV2
import segmentation_models_pytorch as smp

loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)

os.makedirs("./data/snapshots/", exist_ok=True)
os.makedirs("./data/visualize/train/", exist_ok=True)
os.makedirs("./data/visualize/test/", exist_ok=True)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data2.to(device), data1.to(device), target.to(device)
        optimizer.zero_grad()
        output1 = model(data1)
        output2 = model(data2)
        loss = loss_Softmax(output1, target) + loss_Softmax(output2, target) + nn.MSELoss()(data1, data2)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
    input_norm = data1[0:8] * 0.5 + 0.5 #[-1, 1] -> [0, 1]
    output_norm = F.softmax(output1, dim=1)[0:8, 1:2].repeat(1, 3, 1, 1).float()
    target_norm = target[0:8].unsqueeze(1).repeat(1, 3, 1, 1).float()

    torchvision.utils.save_image(
        torch.cat([input_norm, output_norm, target_norm], dim=0),
        # f"./data/visualize/train/epoch_{epoch}_idx_{batch_idx}.jpg", 
        f"./data/visualize/train/latest.jpg", 
        normalize=True, range=(0, 1), nrow=len(input_norm), padding=10, pad_value=0.5
    )
    
    torch.save(
        model.state_dict(), 
        # f"./data/snapshots/epoch_{epoch}_idx_{batch_idx}.pt",
        f"./data/snapshots/latest.pt",
    )

best = 0.9743
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    iou = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_Softmax(output, target).item()  # sum up batch loss
            
            input_norm = data * 0.5 + 0.5 #[-1, 1] -> [0, 1]
            output_norm = F.softmax(output, dim=1)[:, 1:2].repeat(1, 3, 1, 1).float()
            target_norm = target.unsqueeze(1).repeat(1, 3, 1, 1).float()
            torchvision.utils.save_image(
                torch.cat([input_norm, output_norm, target_norm], dim=0),
                f"./data/visualize/test/latest_{batch_idx}.jpg", 
                normalize=True, range=(0, 1), nrow=len(input_norm), padding=10, pad_value=0.5
            )
            
            pred = output_norm>0.5
            gt = target_norm>0.5
            correct += pred.eq(gt).sum().item() / gt.numel()
            iou += ((pred & gt).sum()+1e-6) / ((pred | gt).sum()+1e-6)

    test_loss /= len(test_loader.dataset) / pred.size(0)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%, IOU: {:.4f}\n'.format(
        test_loss,
        100. * correct / len(test_loader.dataset) * pred.size(0),
        iou / len(test_loader.dataset) * pred.size(0)
    ))

    global best
    if iou > best:
        best = iou
        torch.save(
            model.state_dict(), 
            f"./data/snapshots/best-{(iou*100): .2f}.pt",
        )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=20.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        Dataset(input_size=256, train=True,
                image_dir="/home/ICT2000/rli/local/data/LIP/ATR/humanparsing/JPEGImages/", 
                label_dir="/home/ICT2000/rli/local/data/LIP/ATR/humanparsing/RemoveBG/"),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Dataset(input_size=256, train=False, 
                image_dir="./data/alignment", 
                label_dir="./data/alignment"),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=2).to(device)

    if os.path.exists("./data/snapshots/latest.pt"):
        model.load_state_dict(
            torch.load(f"./data/snapshots/latest.pt")
        )
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()