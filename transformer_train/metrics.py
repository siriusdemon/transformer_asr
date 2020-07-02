import torch
from torch import nn
import torch.nn.functional as F
#from otrans.data import PAD

PAD=1
class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, smoothing, padding_idx=PAD, normalize_length=True,
                 criterion=nn.KLDivLoss(reduction='none')):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.reshape(-1)
#        print(x.shape,target.shape,91)
        with torch.no_grad():
            
            true_dist = x.clone()
#            print(true_dist.shape,true_dist,80)
            true_dist.fill_(self.smoothing / (self.size - 1))
#            print(true_dist,81)
            ignore = target == self.padding_idx  # (B,)
#            print(80,ignore.shape,ignore)
            total = len(target) - ignore.sum().item()
#            print(81,total)
            target = target.masked_fill(ignore, 0)  # avoid -1 index
#            print(82,target.shape,target)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
#        print(true_dist.shape,true_dist[-1][:50],93)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
