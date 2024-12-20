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
from losses import FFLoss, SymBaLoss


def wrong(opt,targets_pos):
    targets_neg = targets_pos.clone()
    for c in range(opt.num_class):
        targets_neg[targets_pos==c] = torch.LongTensor(np.random.choice(list(set(np.arange(opt.num_class)) - {c}),(targets_pos==c).sum().item())).cuda()
    return targets_neg

def one_epoch_stage1(loader, model, criterion, optimizers, opt, phase='train'):
    
    losses = torch.zeros(opt.L)
    n      = 0
    
    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)

    for batch in loader:
        x = batch[0].to('cuda')
        
        y_pos = batch[1].to('cuda')
        y_neg = wrong(opt,y_pos)

        n += len(y_pos)
        x_pos = model.patching_layer(x, y_pos)
        x_neg = model.patching_layer(x, y_neg)

        for l in range(opt.L):
            
            x_pos = model.layers[l](x_pos.detach())
            x_neg = model.layers[l](x_neg.detach())
            loss  = criterion(x_pos[:,1:].mean(1),x_neg[:,1:].mean(1))

            if phase=='train':
                optimizers[l].zero_grad()
                loss.backward()
                optimizers[l].step()

            x_pos = F.normalize(torch.flatten(x_pos,1)).view(x_pos.shape)
            x_neg = F.normalize(torch.flatten(x_neg,1)).view(x_neg.shape)

            losses[l] += loss.item() * len(y_pos)

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


def eval_energy(test_loader, model, opt):

    model.eval()

    num_corrects = 0
    n            = 0

    torch.set_grad_enabled(False)
    for batch in test_loader:

        x = batch[0].to('cuda')
        y = batch[1].to('cuda')
        n += len(y)

        # Extracting feature
        g = torch.zeros(len(x), opt.num_class).cuda()
        for c in range(opt.num_class):
            x_ = model.patching_layer(x, torch.ones(len(x)).long().cuda() * c)
            for l in range(opt.L):
                x_ = model.layers[l](x_)
                if l != 0: g[:,c] = x_[:,1:].mean(1).pow(2).mean(1)
                x_ = F.normalize(torch.flatten(x_,1)).view(x_.shape)

        _, pred = g.topk(opt.eval_mode, 1, True, True)
        num_corrects += pred.eq(y.view(-1, 1).expand_as(pred)).reshape(-1).float().sum(0, keepdim=True)

    accuracy = num_corrects/n

    return accuracy



def main():

    opt = parse_option()
    
    # build data loader
    print('\n################## Preparing data ##################\n')
    opt.one_forward = True
    loaders = set_loaders(opt)
    opt.num_patches = opt.num_patches+1
    opt.ff = True

    # build model and criterion
    model = ViT(opt).to('cuda')

    # build optimizer
    optimizers = set_optimizers(model, opt)

    if opt.symba: criterion = SymBaLoss(opt)
    else:         criterion = FFLoss(opt)

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

        losses['train'] = one_epoch_stage1(loaders['train'], model, criterion, optimizers, opt, phase='train')
        losses['valid'] = one_epoch_stage1(loaders['valid'], model, criterion, optimizers, opt, phase='valid')

        time2  = time.time()
        
        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs1 + first_epoch, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'][-1] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'][-1].item() )
            loss_valid_min = losses['valid'][-1].item()
            torch.save(model.state_dict(), './save/model_best.pth')

        save_model(model , optimizers, epoch, loss_valid_min)

    if opt.one_pass_softmax:
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

            print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs2, time2 - time1))

            print(losses['train'])
            print(losses['valid'])

            if losses['valid'] < loss_valid_min:
                print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'])
                loss_valid_min = losses['valid']
                torch.save(model.state_dict(), './save/model_best.pth')
            

    print('\n################## Evaluation ##################\n')
    # Eval
    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))
    time1  = time.time()
    accuracy = eval_energy(loaders['test'], model, opt)
    time2  = time.time()
    print(time2 - time1)
    print(accuracy*100)

if __name__ == '__main__':
    main()