import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class TripletSelHNLoss(nn.Module):

    def __init__(self, opt):
        super(TripletSelHNLoss, self).__init__()
        self.opt = opt
        self.margin = opt.margin
        self.epsilon = opt.epsilon
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size)
        if torch.cuda.is_available():
            self.pos_mask = self.pos_mask.cuda()

    def max_violation_on(self):
        pass

    def max_violation_off(self):
        pass

    def forward(self, v, t):
        '''
        scores are the similarities between all samples,
        the values on the diagonal are the similarities of the positive pairs,
        the value on the off-diagonal are the similarities of the negative pairs
        '''

        scores = get_sim(v, t)
        pos_scores = scores.diag()
        diagonal = pos_scores.view(scores.size(0), 1)
        pos_scores_1 = diagonal.expand_as(scores)
        pos_scores_2 = diagonal.t().expand_as(scores)

        if scores.size(0) != self.batch_size:
            pos_mask = torch.eye(scores.size(0))
            if torch.cuda.is_available():
                pos_mask = pos_mask.cuda()
        else:
            pos_mask = self.pos_mask
        neg_mask = 1 - pos_mask

        # mean
        mean_loss_1 = (scores - pos_scores_1 + self.margin).clamp(min=0)
        mean_loss_2 = (scores - pos_scores_2 + self.margin).clamp(min=0)
        mean_loss_1 = mean_loss_1 * neg_mask
        mean_loss_2 = mean_loss_2 * neg_mask
        mean_loss_1 = mean_loss_1.mean(1)
        mean_loss_2 = mean_loss_2.mean(0)

        # hard negative
        neg_scores = scores * neg_mask - pos_mask
        neg_scores_1 = neg_scores.max(1)[0]
        neg_scores_2 = neg_scores.max(0)[0]

        loss_1 = (neg_scores_1 - pos_scores_1 + self.margin).clamp(min=0)
        loss_2 = (neg_scores_2 - pos_scores_2 + self.margin).clamp(min=0)

        loss_1 = torch.where(torch.abs(pos_scores - neg_scores_1) < self.epsilon, mean_loss_1, loss_1)
        loss_2 = torch.where(torch.abs(pos_scores - neg_scores_2) < self.epsilon, mean_loss_2, loss_2)

        loss_1 = loss_1.mean()
        loss_2 = loss_2.mean()
        loss = loss_1 + loss_2

        return loss


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

