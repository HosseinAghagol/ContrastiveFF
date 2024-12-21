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
        self.one_forward = opt.one_forward

    def forward(self, features, labels=None):
        device = 'cuda'

        labels = labels.contiguous().view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)

        n = features[0].shape[0]
        count = 1 if self.one_forward else 2

        contrast_feature = F.normalize(torch.cat(features, dim=0))
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

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.where(mask == 1,
                                  torch.clamp(anchor_dot_contrast+self.positive_margin,max=1),
                                  anchor_dot_contrast)

        anchor_dot_contrast = torch.div(anchor_dot_contrast,self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - 1 * mean_log_prob_pos

        loss = loss.view(count, n).mean()

        return loss
    

# class FFLoss(nn.Module):

#     def __init__(self, opt):
#         super(FFLoss, self).__init__()
#         self.threshold = opt.threshold

#     def forward(self, a_pos,a_neg):

#         g_pos = a_pos.pow(2).mean(1)
#         g_neg = a_neg.pow(2).mean(1)
        
#         return torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
    
class FFLoss(nn.Module):
    def __init__(self, opt):
        super(FFLoss, self).__init__()
        self.threshold = opt.threshold

    def forward(self, a_pos, a_neg):
        g_pos = a_pos.pow(2).mean(1)
        g_neg = a_neg.pow(2).mean(1)
        
        # Numerical stability adjustment
        logits = torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
        max_logit = logits.max()
        logits_stable = logits - max_logit  # Stabilize values to avoid large exp values

        return max_logit + torch.log(1 + torch.exp(logits_stable)).mean()
    
class SymBaLoss(nn.Module):

    def __init__(self, opt):
        super(SymBaLoss, self).__init__()
        self.alpha = opt.alpha

    def forward(self, a_pos,a_neg):

        g_pos = a_pos.pow(2).mean(1)
        g_neg = a_neg.pow(2).mean(1)
        return torch.log(1 + torch.exp(-self.alpha*(g_pos - g_neg))).mean()



