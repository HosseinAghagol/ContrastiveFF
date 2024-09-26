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


def one_epoch_stage1(loader, model, criterions, optimizers, opt, phase='train'):
    model.train() if phase=='train' else model.eval()

    losses = torch.zeros(opt.L)
    n      = 0

    torch.set_grad_enabled(True if phase=='train' else False)
    for batch in tqdm(loader[phase]):
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
                loss = criterions[l]([x1,x2], targets)

            if phase=='Train':
                optimizers[l].zero_grad()
                loss.backward()
                optimizers[l].step()

            losses[l] += loss.item() * len(targets)

    return losses/n


def main():

    opt = parse_option()
    
    # build data loader
    loaders = set_loaders(opt)

    # build model and criterion
    model = ViT(opt).to('cuda')

    # build optimizer
    optimizers = set_optimizers(model, opt)
    positive_margin = np.linspace(opt.m0, 0.1, opt.L)
    criterions = [SupMCon(opt, positive_margin[l]) for l in range(len(model.layers))]

    loss_valid_min = -np.inf
    

    # training routine
    ##################### Stage 1 #####################
    for epoch in range(1, opt.epochs + 1):
        losses = {'train':0,'valid':0}
        # train for one epoch
        time1  = time.time()

        losses['train'] = one_epoch_stage1(loaders, model, criterions, optimizers, opt, phase='train')
        losses['valid'] = one_epoch_stage1(loaders, model, criterions, optimizers, opt, phase='valid')

        time2  = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        # if losses['valid'][-1] < loss_valid_min:
        #     save_model(model,optimizers)

    ##################### Stage 2 #####################

if __name__ == '__main__':
    main()