import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data.sampler import SubsetRandomSampler

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--data', type=str, default=10,choices=['cifar10', 'cifar100','tiny_imagenet'], help='set data')
    
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    
    parser.add_argument('--epochs1', type=int, default=600, help='number of training epochs')

    parser.add_argument('--epochs2', type=int, default=50, help='number of training epochs')
    
    parser.add_argument('--E', type=int, default=128, help='')
    
    parser.add_argument('--H', type=int, default=4, help='')
    
    parser.add_argument('--L', type=int, default=6, help='')

    parser.add_argument('--patch_size', type=int, default=4, help='')
    
    parser.add_argument('--lr1', type=float, default=0.004, help='learning rate stage 1')

    parser.add_argument('--lr2', type=float, default=0.0005, help='learning rate stage 2')
    
    parser.add_argument('--temp', type=float, default=0.15, help='temperature for contrastive loss function')

    parser.add_argument('--one_forward', action='store_true', help='')
    
    parser.add_argument('--m0', type=float, default=0.4, help='')
    
    parser.add_argument('--mL', type=float, default=0.1, help='')

    parser.add_argument('--randaug', action='store_true', help='')

    parser.add_argument('--resume', action='store_true', help='')

    parser.add_argument('--non_linear_m', action='store_true', help='')
    

    # parser.add_argument('--print_freq', type=int, default=10,
    #                     help='print frequency')
    # parser.add_argument('--save_freq', type=int, default=50,
    #                     help='save frequency')
    # parser.add_argument('--batch_size', type=int, default=256,
    #                     help='batch_size')
    # parser.add_argument('--num_workers', type=int, default=16,
    #                     help='num of workers to use')
    # parser.add_argument('--epochs', type=int, default=1000,
    #                     help='number of training epochs')

    # # optimization
    # parser.add_argument('--learning_rate', type=float, default=0.05,
    #                     help='learning rate')
    # parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
    #                     help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1,
    #                     help='decay rate for learning rate')
    # parser.add_argument('--weight_decay', type=float, default=1e-4,
    #                     help='weight decay')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum')

    # # model dataset
    # parser.add_argument('--model', type=str, default='resnet50')
    # parser.add_argument('--dataset', type=str, default='cifar10',
    #                     choices=['cifar10', 'cifar100', 'path'], help='dataset')
    # parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    # parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    # parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    # parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # # method
    # parser.add_argument('--method', type=str, default='SupCon',
    #                     choices=['SupCon', 'SimCLR'], help='choose method')

    # # temperature
    # parser.add_argument('--temp', type=float, default=0.07,
    #                     help='temperature for loss function')

    # # other setting
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')
    # parser.add_argument('--syncBN', action='store_true',
    #                     help='using synchronized batch normalization')
    # parser.add_argument('--warm', action='store_true',
    #                     help='warm-up for large batch training')
    # parser.add_argument('--trial', type=str, default='0',
    #                     help='id for recording multiple runs')

    opt = parser.parse_args()


    # set the path according to the environment

    opt.data_folder = './data/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.data)

    opt.model_name = 'CFF_{}_ViT_[{} {} {}]_lr_{}_bsz_{}_temp_{}_m0{}'.\
        format('CFF', opt.data,'ViT',opt.E,opt.H,opt.L, opt.lr1, opt.batch_size, opt.temp, opt.m0)

    if not os.path.isdir('save'):
        os.makedirs('save')
        
    return opt


# class TwoCropTransform:
#     """Create two crops of the same image"""
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, x):
#         return [v2.ToImage()(x), v2.ToImage()(x)]
        
def set_loaders(opt):
    valid_size = 0.1
    train_transform = []
    # if opt.randaug: train_transform.append(v2.RandAugment(2,14))
    # train_transform.extend([v2.RandomCrop(32, padding=4),
    #                         v2.RandomHorizontalFlip(),
    #                         v2.ToDtype(torch.float32, scale=True),
    #                         v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset  = datasets.CIFAR10('./data/',train=True,transform=v2.ToImage(),download=True)
    test_dataset   = datasets.CIFAR10('./data/',train=False,transform=test_transform,download=True)

    opt.patch_size  = 4
    opt.num_patches = int((32**2) / (opt.patch_size**2))
    opt.num_class   = 10
    
    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    indices   = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_idx), pin_memory=True, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(valid_idx), pin_memory=True, sampler=valid_sampler)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    return {'train': train_loader, 'valid':valid_loader, 'test':test_loader}, {'train': train_transform, 'valid':test_transform}

def load_data_on_ram(loader):

    temp = next(iter(loader))

    x = temp[0]
    y = temp[1]

    return x, y

def set_optimizers(model,opt):
    optimizers = []
    for l in range(opt.L):
        optimizers.append(torch.optim.AdamW(model.layers[l].parameters() , lr=opt.lr1))
    return optimizers

def save_model(model, optimizers, epoch, loss_min):
    state = {
        'model': model.state_dict(),
        'optimizer': [optimizers[l].state_dict() for l in range(len(optimizers))],
        'epoch': epoch,
        'loss_min':loss_min
    }
    torch.save(state, './save/cp.pth')


def load_model(model, optimizers):
    # Load the checkpoint
    checkpoint = torch.load('./save/cp.pth', weights_only=True)

    # Restore the model state
    model.load_state_dict(checkpoint['model'])

    # Restore each optimizer's state
    for l in range(len(optimizers)):
        optimizers[l].load_state_dict(checkpoint['optimizer'][l])

    # Restore the epoch number
    epoch = checkpoint['epoch']
    loss_min = checkpoint['loss_min']
    
    return model, optimizers, epoch, loss_min