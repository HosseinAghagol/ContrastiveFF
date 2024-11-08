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

from models.vit import ViT
from losses import SupMCon


def one_epoch_stage1(loader, model, criterions, optimizers, opt, phase='train'):
    

    losses = torch.zeros(opt.L)
    n      = 0
    
    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)

    for batch in loader:
        if opt.one_forward:
            x1 = batch[0].to('cuda')
        else:
            x1, x2 = batch[0][0].to('cuda'), batch[0][1].to('cuda')
        
        targets = batch[1].to('cuda')

        n += len(targets)

        for l in range(opt.L):

            if opt.one_forward:
                x1 = model.layers[l](x1.detach())
                loss = criterions[l]([x1.mean(1)], targets)

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



def one_epoch_stage2(loader, model, criterion, optimizer, opt, phase='train'):
    losses = 0
    n      = 0

    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)
    for batch in loader:
        if opt.one_forward:
            features = batch[0].to('cuda')
        else:
            features = batch[0][0].to('cuda')

        targets = batch[1].to('cuda')

        n += len(targets)

        # Extracting feature
        model.eval()
        with torch.no_grad():
            for l in range(opt.L): features = model.layers[l](features)

        # Classifier head
        model.train() if phase=='train' else model.eval()
        torch.set_grad_enabled(True if phase=='train' else False)
        outputs = model.classifier_head(features.mean(1).detach())
        loss   = criterion(outputs, targets)

        if phase=='train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses += loss.item() * len(targets)

    return losses/n


def eval(test_loader, model, opt):

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
        for l in range(opt.L): features = model.layers[l](features)
        # Classifier head
        output = model.classifier_head(features.mean(1))
        _, pred = output.topk(opt.eval_mode, 1, True, True)
        num_corrects += pred.eq(targets.view(-1, 1).expand_as(pred)).reshape(-1).float().sum(0, keepdim=True)

    accuracy = num_corrects/n

    return accuracy



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
    if opt.non_linear_m:
        positive_margin = opt.mL + (opt.m0 - opt.mL) * (np.exp(-5 * np.linspace(0, 1, opt.L)))
    
    criterions = [SupMCon(opt, positive_margin[l]) for l in range(len(model.layers))]
    if opt.m0 ==0: criterions = [SupMCon(opt, 0) for l in range(len(model.layers))]

    loss_valid_min = np.inf
    
    first_epoch = 0
    if opt.resume:
        model, optimizers, first_epoch, loss_valid_min = load_model(model, optimizers)

    # training routine
    print('\n################## Training-Stage 1 ##################\n')
    # Stage 1 
    for epoch in range(first_epoch+1, opt.epochs1 + first_epoch+1):
        losses = {'train':0,'valid':0}
        # train for one epoch

        time1  = time.time()

        losses['train'] = one_epoch_stage1(loaders['train'], model, criterions, optimizers, opt, phase='train')
        losses['valid'] = one_epoch_stage1(loaders['valid'], model, criterions, optimizers, opt, phase='valid')

        time2  = time.time()
        
        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs1 + first_epoch, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'][-1] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'][-1].item() )
            loss_valid_min = losses['valid'][-1].item()
            torch.save(model.state_dict(), './save/model_best.pth')

        save_model(model , optimizers, epoch, loss_valid_min)


    print('\n################## Training-Stage 2 ##################\n')
    # Stage 2

    
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=opt.lr2)
    criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))


    loss_valid_min = np.inf
    for epoch in range(1, opt.epochs2 + 1):
        losses = {'train':0,'valid':0}

        # train for one epoch
        time1  = time.time()

        losses['train'] = one_epoch_stage2(loaders['train'], model, criterion, optimizer, opt, phase='train')
        losses['valid'] = one_epoch_stage2(loaders['valid'], model, criterion, optimizer, opt, phase='valid')

        time2  = time.time()

        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs2 + first_epoch, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'])
            loss_valid_min = losses['valid']
            torch.save(model.state_dict(), './save/model_best.pth')
            
    print('\n################## Evaluation ##################\n')
    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))
    accuracy = eval(loaders['test'], model, opt)
    print(accuracy*100)

if __name__ == '__main__':
    main()