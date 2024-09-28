import torch
from torch import nn
import torch.nn.functional as F

class SupMCon(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, opt, positive_margin):
        super(SupMCon, self).__init__()
        self.temperature     = opt.temp
        self.positive_margin = positive_margin
        self.one_forward     = opt.one_forward

    def forward(self, features, labels=None):
        device = 'cuda'

        labels = labels.contiguous().view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)

        n = features[0].shape[1]
        count = 2

        contrast_feature = torch.cat(features, dim=0)
    

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
        contrast_feature = F.normalize(contrast_feature)
        anchor_feature   = contrast_feature

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