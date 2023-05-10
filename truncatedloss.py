import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt


class TruncatedLoss(nn.Module):

    def __init__(self, q, k, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
        self.count = 0

    def forward(self, logits, targets, indexes):
        # p = F.softmax(logits, dim=1)
        Yg = torch.gather(logits, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
               self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes, device, flag_noise_type):
        # p = F.softmax(logits, dim=1)
        Yg = torch.gather(logits, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = Lqk.to(device)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        condition = torch.gt(Yg, self.k)
        condition = torch.gt(Lq, self.k)

        noise_indices = np.where(flag_noise_type == 1)[0]

        # plt.stem(indexes, Lq.detach().cpu().numpy(), linefmt='b-', markerfmt='bo', label='Normal')
        # plt.stem(indexes[noise_indices], Lq[noise_indices].detach().cpu().numpy(), linefmt='r-', markerfmt='ro', label='Noise')

        plt.stem(indexes, Lq.detach().cpu(), markerfmt='bo', label='Normal')
        plt.stem(indexes[noise_indices], Lq[noise_indices].detach().cpu(), markerfmt='ro', label='Noise')

        plt.xlabel('Index')
        plt.ylabel('Lq')
        plt.title('Lq')
        plt.legend()
        plt.savefig(f'Lq_plot_{self.count}.png')
        self.count += 1

        temp = condition.type(torch.cuda.FloatTensor)
        self.weight[indexes] = temp.to(device)

    def get_weight(self, indexes):
        return self.weight[indexes].squeeze()

