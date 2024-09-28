import torch
from torch import nn
import torch.nn.functional as F





class SupMCon(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, opt, positive_margin, contrast_mode='all',
                 base_temperature=0.07):
        super(SupMCon, self).__init__()
        self.temperature = opt.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.positive_margin = positive_margin

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)



        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        # compute logits
        anchor_feature = F.normalize(anchor_feature)
        contrast_feature = F.normalize(contrast_feature)

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.where(mask == 1,
                                  torch.clamp(anchor_dot_contrast+self.positive_margin,max=1),
                                  anchor_dot_contrast)


        anchor_dot_contrast = torch.div(anchor_dot_contrast,self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast - 0



        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - 1 * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class SupMCon1(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, opt, positive_margin):
        super(SupMCon1, self).__init__()
        self.temperature     = opt.temp
        self.positive_margin = positive_margin

    def forward(self, features, labels=None):
        device = 'cuda'

        labels = labels.contiguous().view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)

        n = features[0].shape[0]
        count = 2

        contrast_feature = torch.cat(features, dim=0)
        anchor_feature = contrast_feature

        # tile mask
        mask = mask.repeat(count, count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        # compute logits
        anchor_feature   = F.normalize(anchor_feature)
        contrast_feature = F.normalize(contrast_feature)

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.where(mask == 1,
                                  torch.clamp(anchor_dot_contrast+self.positive_margin,max=1),
                                  anchor_dot_contrast)

        # anchor_dot_contrast = torch.where(mask == 0,
        #                           torch.clamp(anchor_dot_contrast-self.negative_margin,min=0),
        #                           anchor_dot_contrast)

        anchor_dot_contrast = torch.div(anchor_dot_contrast,self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - 1 * mean_log_prob_pos

        loss = loss.view(count, n).mean()

        return loss