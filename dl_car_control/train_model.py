from __future__ import print_function
from dataset_test import *
from pilotnet import *
import argparse
import torch
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the cucriterionrrent Model')
   
    args = parser.parse_args()

    use_cuda = args.use_cuda

    training_data = CustomImageDataset('output_simple.csv','dataset_simple')

    trainloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0' if (use_cuda and torch.cuda.is_available()) else 'cpu')

    model = PilotNet((3,60,200), 2).to(device)


    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainloader, optimizer, epoch)
        scheduler.step()

    print('Finished Training')
    
    if use_cuda:
        dummy_input = torch.randn(1, 3, 60, 200,device=torch.device("cuda"))
    else:
        dummy_input = torch.randn(1, 3, 60, 200)        

    #torch.save(model.state_dict(), "mynet.pt")
    torch.onnx.export(model, dummy_input, "mynet.onnx", verbose=True, export_params=True, opset_version=9, input_names=['input'], output_names=['output'])








    

# Execute!
if __name__ == "__main__":
    main()