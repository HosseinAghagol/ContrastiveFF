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
# from utils import save_model
from utils import set_loaders

from models.vit import ViT
from losses import SupMCon


def one_epoch_stage1(loaders, model, criterions, optimizers, opt, phase='train'):
    model.train() if phase=='train' else model.eval()

    losses = torch.zeros(opt.L)
    n      = 0

    torch.set_grad_enabled(True if phase=='train' else False)
    for batch in loaders[phase]:
        if opt.one_forward: x1 = batch[0].to('cuda')
        else: x1,x2 = batch[0][0].to('cuda'),batch[0][1].to('cuda')

        targets = batch[1].to('cuda')

        n += len(targets)

        for l in range(opt.L):

            if opt.one_forward:
                x1 = model.layers[l](x1.detach())
                loss = criterions[l]([x1], targets)

            else: 
                x1 = model.layers[l](x1.detach())
                x2 = model.layers[l](x2.detach())
                loss = criterions[l]([x1.mean(1),x2.mean(1)], targets)


            if phase=='train':
                optimizers[l].zero_grad()
                loss.backward()
                optimizers[l].step()

            losses[l] += loss.item() * len(targets)
    return losses/n



def one_epoch_stage2(loaders, model, criterion, optimizer, opt, phase='train'):
    model.train() if phase=='train' else model.eval()

    losses = 0
    n      = 0

    
    for batch in loaders[phase]:

        features = batch[0][0].to('cuda')
        targets  = batch[1].to('cuda')

        n += len(targets)

        # Extracting feature
        model.eval()
        with torch.no_grad():
            for l in range(opt.L): features = model.layers[l](features)

        # Classifier head
        model.train() if phase=='train' else model.eval()
        torch.set_grad_enabled(True if phase=='train' else False)
        outputs = model.classifier_head(features.detach())
        print(outputs.shape)
        loss   = criterion(outputs, targets)

        if phase=='train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item() * len(targets)

    return losses/n



def eval(test_loader, model, criterion, opt):

    model.eval()
    losses = 0
    n      = 0

    torch.set_grad_enabled(False)
    for batch in test_loader:

        features = batch[0].to('cuda')
        targets  = batch[1].to('cuda')

        n += len(targets)

        # Extracting feature
        for l in range(opt.L): features = model.layers[l](features)
        # Classifier head
        output = model.classifier_head(features.detach())
        loss   = criterion(output, targets)
        num_corrects = torch.sum(torch.argmax(output, dim=1) == targets).cpu().item()

    accuracy = num_corrects/n

    return losses/n, accuracy



def main():

    opt = parse_option()
    
    # build data loader
    print('\n################## Preparing data ##################\n')
    loaders = set_loaders(opt)

    # build model and criterion
    model = ViT(opt).to('cuda')

    # build optimizer
    optimizers = set_optimizers(model, opt)
    positive_margin = np.linspace(opt.m0, opt.mL, opt.L)
    criterions = [SupMCon(opt, positive_margin[l]) for l in range(len(model.layers))]

    loss_valid_min = -np.inf
    

    # training routine
    print('\n################## Training-Stage 1 ##################\n')
    # Stage 1 
    for epoch in range(1, opt.epochs1 + 1):
        losses = {'train':0,'valid':0}
        # train for one epoch
        time1  = time.time()

        losses['train'] = one_epoch_stage1(loaders, model, criterions, optimizers, opt, phase='train')
        losses['valid'] = one_epoch_stage1(loaders, model, criterions, optimizers, opt, phase='valid')

        time2  = time.time()

        print('epoch [{},{}], {:.2f}'.format(epoch, opt.epochs1, time2 - time1))

        print()
        print(losses['train'])
        print(losses['valid'])
        print()
        
        # if losses['valid'][-1] < loss_valid_min:
        #     save_model(model,optimizers)


    print('\n################## Training-Stage 2 ##################\n')
    # Stage 2

    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=opt.lr2)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.epochs2 + 1):
        loss = {'train':0,'valid':0}
        # train for one epoch
        time1  = time.time()

        loss['train'] = one_epoch_stage2(loaders, model, criterion, optimizer, opt, phase='train')
        loss['valid'] = one_epoch_stage2(loaders, model, criterion, optimizer, opt, phase='valid')

        time2  = time.time()

        print('epoch [{},{}], {:.2f}'.format(epoch, opt.epochs2, time2 - time1))

        print()
        print(loss['train'])
        print(loss['valid'])
        print()

    print('\n################## Evaluation ##################\n')

    loss, accuracy = eval(loaders['test'], model, criterion, opt)
    print(loss, accuracy)

if __name__ == '__main__':
    main()