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
from utils import set_loaders, load_data_on_ram

from models.vit import ViT
from losses import SupMCon


def one_epoch_stage1(x, y, transforms, model, criterions, optimizers, opt, phase='train'):
    

    losses = torch.zeros(opt.L)
    n      = 0
    
    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)
    for i in range(len(y)//opt.batch_size+1):

        if opt.one_forward:
            x1 = transforms[phase](x[i*opt.batch_size:(i+1)*opt.batch_size])
        else:
            x1, x2 = transforms[phase](x[i*opt.batch_size:(i+1)*opt.batch_size]).to('cuda'), transforms[phase](x[i*opt.batch_size:(i+1)*opt.batch_size]).to('cuda')

        targets = y[i*opt.batch_size:(i+1)*opt.batch_size].to('cuda')


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



def one_epoch_stage2(x, y, transforms, model, criterion, optimizer, opt, phase='train'):
    losses = 0
    n      = 0

    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)
    for i in range(len(y)//opt.batch_size+1):

        features = transforms[phase](x[i*opt.batch_size:(i+1)*opt.batch_size]).to('cuda')
        targets  = y[i*opt.batch_size:(i+1)*opt.batch_size].to('cuda')

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



def eval(test_loader, model, criterion, opt):

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

        loss   = criterion(output, targets)
        
        num_corrects += torch.sum(torch.argmax(output, dim=1) == targets).cpu().item()
        losses += loss.item() * len(targets)

    accuracy = num_corrects/n

    return losses/n, accuracy



def main():

    opt = parse_option()
    
    # build data loader
    print('\n################## Preparing data ##################\n')
    loaders, transforms = set_loaders(opt)
    
    x_train,y_train = load_data_on_ram(loaders['train'])
    x_valid,y_valid = load_data_on_ram(loaders['valid'])

    # build model and criterion
    model = ViT(opt).to('cuda')

    # build optimizer
    optimizers = set_optimizers(model, opt)
    positive_margin = np.linspace(opt.m0, opt.mL, opt.L)
    criterions = [SupMCon(opt, positive_margin[l]) for l in range(len(model.layers))]

    loss_valid_min = np.inf
    
    first_epoch = 0
    if opt.resume:
        model, optimizers, first_epoch, loss_valid_min = load_model(model, optimizers)

    # training routine
    print('\n################## Training-Stage 1 ##################\n')
    # Stage 1 
    for epoch in range(first_epoch+1, opt.epochs1 + first_epoch):
        losses = {'train':0,'valid':0}
        # train for one epoch

        indices = torch.randperm(len(y_train))
        x_train, y_train = x_train[indices], y_train[indices]
        indices = torch.randperm(len(y_valid))
        x_valid, y_valid = x_valid[indices], y_valid[indices]

        time1  = time.time()

        losses['train'] = one_epoch_stage1(x_train, y_train, transforms, model, criterions, optimizers, opt, phase='train')
        losses['valid'] = one_epoch_stage1(x_valid, y_valid, transforms, model, criterions, optimizers, opt, phase='valid')

        time2  = time.time()
        
        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs1 + first_epoch-1, time2 - time1))

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

        indices = torch.randperm(len(y_train))
        x_train, y_train = x_train[indices], y_train[indices]
        indices = torch.randperm(len(y_valid))
        x_valid, y_valid = x_valid[indices], y_valid[indices]

        # train for one epoch
        time1  = time.time()

        losses['train'] = one_epoch_stage2(x_train, y_train, transforms, model, criterion, optimizer, opt, phase='train')
        losses['valid'] = one_epoch_stage2(x_valid, y_valid, transforms, model, criterion, optimizer, opt, phase='valid')

        time2  = time.time()

        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs1 + first_epoch-1, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'])
            loss_valid_min = losses['valid']
            torch.save(model.state_dict(), './save/model_best.pth')
            
    print('\n################## Evaluation ##################\n')
    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))
    losses, accuracy = eval(loaders['test'], model, criterion, opt)
    print(losses, accuracy)

if __name__ == '__main__':
    main()