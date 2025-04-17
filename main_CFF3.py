import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import parse_option
from utils import set_optimizers
from utils import save_model,load_model
from utils import set_loaders
from utils import set_margins

from models.maxvit import MaxViT
from losses import SupMCon


def one_epoch_stage1(loader, model, criterions, optimizers, args, phase='train'):
    

    losses = torch.zeros(args.L)
    n      = 0
    
    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)

    for batch in loader:

        x1, x2  = batch[0][0].to('cuda'), batch[0][1].to('cuda')
        targets = batch[1].to('cuda')
        n += len(targets)
        for l in range(args.L):
            x1 = model.layers[l](x1.detach())
            x2 = model.layers[l](x2.detach())
            loss = criterions[l]([x1.mean([2,3]),x2.mean([2,3])], targets)

            if phase=='train':
                optimizers[l].zero_grad()
                loss.backward()
                optimizers[l].step()

            losses[l] += loss.item() * len(targets)

    return losses/n



def one_epoch_stage2(loader, model, criterion, optimizer, args, phase='train'):
    losses = 0
    n      = 0

    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)
    for batch in loader:

        features = model.batch[0][0].to('cuda')
        targets  = batch[1].to('cuda')
        n += len(targets)

        # Extracting feature
        model.eval()
        with torch.no_grad():
            for l in range(args.L): features = model.layers[l](features)

        # Classifier head
        model.train() if phase=='train' else model.eval()
        torch.set_grad_enabled(True if phase=='train' else False)
        outputs = model.classifier_head(features.mean([2,3]).detach())
        loss   = criterion(outputs, targets)

        if phase=='train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses += loss.item() * len(targets)

    return losses/n


def eval(test_loader, model, args):

    model.eval()

    losses       = 0
    num_corrects = 0
    n            = 0

    torch.set_grad_enabled(False)
    for batch in test_loader:

        features = batch[0].to('cuda')
        targets  = batch[1].to('cuda')
        n += len(targets)

        # Extracting feature
        for l in range(args.L): features = model.layers[l](features)
        # Classifier head
        output = model.classifier_head(features.mean([2,3]))
        _, pred = output.topk(args.eval_mode, 1, True, True)
        num_corrects += pred.eq(targets.view(-1, 1).expand_as(pred)).reshape(-1).float().sum(0, keepdim=True)

    accuracy = num_corrects/n

    return accuracy



def main():

    args = parse_option()
    args.one_forward = False
    # build data loader
    print('\n################## Preparing data ##################\n')
    loaders = set_loaders(args)

    # build model and criterion
    model = MaxViT(
                    num_classes = 1000,
                    dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
                    dim = 96,                         # dimension of first layer, doubles every layer
                    dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
                    depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
                    window_size = 7,                  # window size for block and grids
                    mbconv_expansion_rate = 4,        # expansion rate of MBConv
                    mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
                    dropout = 0.1                     # dropout
                ).to('cuda')
    # model = nn.DataParallel(model)
    args.L = len(model.layers)

    # build optimizer
    optimizers = set_optimizers(model, args)
    positive_margin = set_margins(args)

    criterions = [SupMCon(args, positive_margin[l]) for l in range(len(model.layers))]
    if args.m0 ==0: criterions = [SupMCon(args, 0) for l in range(len(model.layers))]

    loss_valid_min = np.inf
    
    first_epoch = 0
    if args.resume:
        model, optimizers, first_epoch, loss_valid_min = load_model(model, optimizers)

    # training routine
    print('\n################## Training-Stage 1 ##################\n')
    # Stage 1 
    for epoch in range(first_epoch+1, args.epochs1 + first_epoch+1):
        losses = {'train':0,'valid':0}
        # train for one epoch

        time1  = time.time()

        losses['train'] = one_epoch_stage1(loaders['train'], model, criterions, optimizers, args, phase='train')
        losses['valid'] = one_epoch_stage1(loaders['valid'], model, criterions, optimizers, args, phase='valid')

        time2  = time.time()
        
        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, args.epochs1 + first_epoch, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'][-1] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'][-1].item() )
            loss_valid_min = losses['valid'][-1].item()
            torch.save(model.state_dict(), './save/model_best.pth')

        save_model(model , optimizers, epoch, loss_valid_min)


    print('\n################## Training-Stage 2 ##################\n')
    # Stage 2

    
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=args.lr2)
    criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))


    loss_valid_min = np.inf
    for epoch in range(1, args.epochs2 + 1):
        losses = {'train':0,'valid':0}

        # train for one epoch
        time1  = time.time()

        losses['train'] = one_epoch_stage2(loaders['train'], model, criterion, optimizer, args, phase='train')
        losses['valid'] = one_epoch_stage2(loaders['valid'], model, criterion, optimizer, args, phase='valid')

        time2  = time.time()

        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, args.epochs2, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'])
            loss_valid_min = losses['valid']
            torch.save(model.state_dict(), './save/model_best.pth')
            
    print('\n################## Evaluation ##################\n')
    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))
    time1  = time.time()
    accuracy = eval(loaders['test'], model, args)
    time2  = time.time()
    print(time2 - time1)
    print(accuracy*100)

if __name__ == '__main__':
    main()
