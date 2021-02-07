import torch
from torch import nn
import pdb

device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')

class SMCL(nn.Module):
    def __init__(self, delta=5.0, r=0.01):
        super(SMCL, self).__init__()
        self.delta = delta # coefficient for mmcl
        self.gamma = r         # hard negative mining ratio

    def forward(self, inputs, targets, is_vec=False):
        m, n = inputs.size()

        if is_vec:
            multiclass = targets
        else:
            targets = torch.unsqueeze(targets, 1)
            multiclass = torch.zeros(inputs.size()).cuda()
            multiclass.scatter_(1, targets, float(1))

        loss = []
        num_pos = 0
        num_hard_neg = 0
        for i in range(m):
            logit = inputs[i].to(device_0)
            label = multiclass[i].to(device_0)

            pos_logit = torch.masked_select(logit, label > 0.5)
            neg_logit = torch.masked_select(logit, label < 0.5)

            _, idx = torch.sort(neg_logit.detach().clone(), descending=True)
            num = int(self.gamma * neg_logit.size(0))
            mask = torch.zeros(neg_logit.size(), dtype=torch.bool).cuda()
            mask[idx[0:num]] = 1
            hard_neg_logit = torch.masked_select(neg_logit, mask)
            num_pos += len(pos_logit)
            num_hard_neg += num
            #print('portion of [total length - %d] pos:neg = %d/%d'%(len(inputs[i]),len(pos_logit),num))

            l = self.delta * torch.mean((pos_logit-1).pow(2)) + torch.mean((hard_neg_logit+1).pow(2))
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        #print('portion of pos:neg = %.3f / %.3f'%(num_pos/m,num_hard_neg/m))
        #return loss, num_pos/m # to eval MLP methods
        return loss

