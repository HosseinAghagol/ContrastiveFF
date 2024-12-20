import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data.sampler import SubsetRandomSampler
import os
import urllib.request
import zipfile
from random import shuffle
from math import floor

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--data', type=str, default='cifar10',choices=['cifar10', 'cifar100','tiny_imagenet'], help='set data')
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
    parser.add_argument('--on_ram', action='store_true', help='')
    parser.add_argument('--trial', type=int, default=1, help='')
    parser.add_argument('--threshold', type=int, default=2, help='')
    parser.add_argument('--symba', action='store_true', help='')
    parser.add_argument('--alpha', type=int, default=1, help='')
    parser.add_argument('--one_pass_softmax', action='store_true', help='')
    
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

    args = parser.parse_args()


    # set the path according to the environment

    args.data_folder = './data/'
    args.model_path = './save/SupCon/{}_models'.format(args.data)

    args.model_name = 'CFF_{}_ViT_[{} {} {}]_lr_{}_bsz_{}_temp_{}_m0{}'.\
        format('CFF', args.data,'ViT',args.E,args.H,args.L, args.lr1, args.batch_size, args.temp, args.m0)

    if not os.path.isdir('save'):
        os.makedirs('save')
        
    return args

def tiny_imagenet():
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    path = "%s/tiny-imagenet-200.zip" % os.getcwd()
    urllib.request.urlretrieve(url, path)
    path_to_zip_file = "%s/tiny-imagenet-200.zip" % os.getcwd()
    directory_to_extract_to = os.getcwd()
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    val_dir = "%s/tiny-imagenet-200/val" % os.getcwd()
    val_annotations = "%s/val_annotations.txt" % val_dir
    val_dict = {}
    with open(val_annotations, 'r') as f:
        for line in f:
            line = line.strip().split()
            assert(len(line) == 6)
            wnind = line[1]
            img_name = line[0]
            boxes = '\t'.join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((img_name, boxes))
    assert(len(val_dict) == 200)
    for wnind, entries in val_dict.items():
        val_wnind_dir = "%s/%s" % (val_dir, wnind)
        val_images_dir = "%s/images" % val_dir
        val_wnind_images_dir = "%s/images" % val_wnind_dir
        os.mkdir(val_wnind_dir)
        os.mkdir(val_wnind_images_dir)
        wnind_boxes = "%s/%s_boxes.txt" % (val_wnind_dir, wnind)
        f = open(wnind_boxes, "w")
        for img_name, box in entries:
            source = "%s/%s" % (val_images_dir, img_name)
            dst = "%s/%s" % (val_wnind_images_dir, img_name)
            os.system("cp %s %s" % (source, dst))
            f.write("%s\t%s\n" % (img_name, box))
        f.close()
    os.system("rm -rf %s" % val_images_dir)

     
class CustomTensorDataset(Dataset):
    def __init__(self, X, y, transform, one_forward):
        self.X = X
        self.y = y
        self.transform = transform
        self.one_forward = one_forward

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image, label = self.X[idx], self.y[idx]

        # Convert tensor to PIL Image for applying transformations
        if self.one_forward:
            image = self.transform(image)
        else:
            image = [self.transform(image), self.transform(image)]

        return image, label
    

def wrong(labels):
    labels_wrong = labels.clone()
    for c in range(10):
        labels_wrong[labels==c] = torch.LongTensor(np.random.choice(list(set(np.arange(10)) - {c}),(labels==c).sum().item()))
    return labels_wrong
        
def set_loaders(args):



    if args.data=='cifar10':

        train_transform = []
        if args.randaug: train_transform.append(v2.RandAugment(3,14))

        train_transform.extend([v2.RandomCrop(32, padding=4),
                                v2.RandomHorizontalFlip(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose([v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
        train_dataset = datasets.CIFAR10('./data/',train=True,transform=v2.ToImage(),download=True)
        test_dataset  = datasets.CIFAR10('./data/',train=False,transform=v2.ToImage(),download=True)

        args.patch_size  = 4
        args.num_patches = int((32**2) / (args.patch_size**2))
        args.num_class   = 10
        args.eval_mode   = 1

    elif args.data=='cifar100':
        train_transform = []
        if args.randaug: train_transform.append(v2.RandAugment(3,14))

        train_transform.extend([v2.RandomCrop(32, padding=4),
                                v2.RandomHorizontalFlip(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose([v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        train_dataset = datasets.CIFAR100('./data/',train=True,transform=v2.ToImage(),download=True)
        test_dataset  = datasets.CIFAR100('./data/',train=False,transform=v2.ToImage(),download=True)

        args.patch_size  = 4
        args.num_patches = int((32**2) / (args.patch_size**2))
        args.num_class   = 100
        args.eval_mode   = 5
       
    elif args.data=='tiny_imagenet':
        train_transform = []
        if args.randaug: train_transform.append(v2.RandAugment(3,14))

        train_transform.extend([v2.RandomCrop(64, padding=6),
                                v2.RandomHorizontalFlip(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose([v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tiny_imagenet()
        train_dataset  = datasets.ImageFolder(root='tiny-imagenet-200/train',transform=v2.ToImage())
        test_dataset   = datasets.ImageFolder(root='tiny-imagenet-200/val'  ,transform=v2.ToImage())

        args.patch_size  = 8
        args.num_patches = int((64**2) / (args.patch_size**2))
        args.num_class   = 200
        args.eval_mode   = 10
         
    

    if args.on_ram:
        print('loading data on ram')
        train_data   = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
        test_data    = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_labels  = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
        torch.manual_seed(args.trial)
        indices = torch.randperm(len(train_data))
        indices_train = indices[:int(0.9 * len(train_data))]
        indices_valid = indices[int(0.9 * len(train_data)):]

        valid_data   = train_data[indices_valid]
        train_data   = train_data[indices_train]
        
        valid_labels = train_labels[indices_valid]
        train_labels = train_labels[indices_train]
        ######################################
        # indices = torch.randperm(int(len(train_data)*0.5))
        # train_data   = train_data[indices]
        # train_labels = train_labels[indices]
        # train_labels[indices] = wrong(train_labels[indices])

        # indices = torch.randperm(int(len(valid_data)*0.5))
        # valid_data   = valid_data[indices]
        # valid_labels = valid_labels[indices]
        # valid_labels[indices] = wrong(valid_labels[indices])
        #######################################
        train_dataset = CustomTensorDataset(train_data, train_labels, transform=train_transform, one_forward=args.one_forward)
        valid_dataset = CustomTensorDataset(valid_data, valid_labels, transform=train_transform, one_forward=args.one_forward)
        test_dataset  = CustomTensorDataset(test_data, test_labels, transform=test_transform, one_forward=True)




    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return {'train': train_loader, 'valid':valid_loader, 'test':test_loader}

def set_optimizers(model,args):
    optimizers = []
    for l in range(args.L):
        optimizers.append(torch.optim.Adam(model.layers[l].parameters() , lr=args.lr1))
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