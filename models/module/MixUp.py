# @Time : 2025/3/27 16:06
# @Author : Zeyong Ji
import torch
import numpy as np
class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam