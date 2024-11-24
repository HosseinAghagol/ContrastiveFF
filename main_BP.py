import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import parse_option
from utils import save_model,load_model
from utils import set_loaders

from models.vit import ViT


def one_epoch(loader, model, criterion, optimizer, phase='train'):
    losses = 0
    n      = 0

    model.train() if phase=='train' else model.eval()
    torch.set_grad_enabled(True if phase=='train' else False)
    for batch in loader:

        features, targets = batch[0].to('cuda'), batch[1].to('cuda')
        n += len(targets)

        outputs = model.classifier(features)
        loss    = criterion(outputs, targets)

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
        # Classifier head
        output = model(features)
        _, pred = output.topk(opt.eval_mode, 1, True, True)
        num_corrects += pred.eq(targets.view(-1, 1).expand_as(pred)).reshape(-1).float().sum(0, keepdim=True)

    accuracy = num_corrects/n

    return accuracy



def main():

    opt = parse_option()
    opt.one_forward = True
    # build data loader
    print('\n################## Preparing data ##################\n')
    loaders = set_loaders(opt)

    # build model and criterion
    model = ViT(opt).to('cuda')
    loss_valid_min = np.inf
    
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=opt.lr2)
    criterion = torch.nn.CrossEntropyLoss()

    first_epoch = 0
    if opt.resume:
        model, optimizers, first_epoch, loss_valid_min = load_model(model, optimizers)


    loss_valid_min = np.inf
    for epoch in range(first_epoch+1, opt.epochs1 + first_epoch+1):
        losses = {'train':0,'valid':0}

        # train for one epoch
        time1  = time.time()

        losses['train'] = one_epoch(loaders['train'], model, criterion, optimizer, phase='train')
        losses['valid'] = one_epoch(loaders['valid'], model, criterion, optimizer, phase='valid')

        time2  = time.time()

        print('----------------------------------------\nepoch [{}/{}], {:.1f}s\n'.format(epoch, opt.epochs1 + first_epoch, time2 - time1))

        print(losses['train'])
        print(losses['valid'])

        if losses['valid'] < loss_valid_min:
            print("\nbest val loss:",loss_valid_min,'---------->',losses['valid'])
            loss_valid_min = losses['valid']
            torch.save(model.state_dict(), './save/model_best.pth')

        save_model(model , [optimizer], epoch, loss_valid_min)

    print('\n################## Evaluation ##################\n')
    model.load_state_dict(torch.load('./save/model_best.pth', weights_only=True))
    accuracy = eval(loaders['test'], model, opt)
    print(accuracy*100)

if __name__ == '__main__':
    main()