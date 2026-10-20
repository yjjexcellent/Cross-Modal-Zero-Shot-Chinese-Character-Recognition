import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ClipInfoCELoss(_Loss):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()

    def forward(self, logits, ground_truth):
       # labels = torch.arange(len(logits)).to(device)
       # print('labels:\n', labels.size(), '\n', labels)
       # print('logits:\n', logits.size(), '\n', logits)
       loss_i = F.cross_entropy(logits, ground_truth)
       loss_t = F.cross_entropy(logits.t(), ground_truth)
       loss = (loss_i+loss_t)/2
       # print('loss_i:', loss_i, '\nloss_t:', loss_t, '\ntotal_loss:', loss)
       return loss


def pixel_loss(out, target):
    loss = F.l1_loss(out, target, reduction="mean")
    # loss = F.mse_loss(out, target)
    return loss


def gan_g_loss(fake_font, fake_uni):
    g_loss = -(fake_font.mean() + fake_uni.mean())
    return g_loss


def gan_d_loss(real_font, real_uni, fake_font, fake_uni):
    d_loss = (F.relu(1. - real_font).mean() + F.relu(1. + fake_font).mean()) + \
             F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()
    return d_loss
