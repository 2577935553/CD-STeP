import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CELoss', 'DiceLoss']
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, target, weight):
        b, c, h, w = x.size()
        p_i = F.softmax(x, dim=1)
        label = F.one_hot(target, num_classes=c).permute([0, 3, 1, 2])
        loss = label * torch.log(p_i + 1e-7)
        loss = -loss.flatten(1, -1).sum(1)
        loss = sum(loss * weight)
        loss /= sum(weight + 1e-5) * h * w

        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, target):
        smooth = 1e-7
        num_classes = x.shape[1]
        x = F.softmax(x, dim=1)
        x = torch.argmax(x, dim=1)
        dsc = torch.zeros(num_classes - 1, dtype=torch.float32)

        for i in range(1, num_classes):
            A = (x == i).to(torch.float32)
            B = (target == i).to(torch.float32)
            intersection = torch.sum(A * B)
            dsc[i - 1] = (2. * intersection) / (torch.sum(A) + torch.sum(B) + smooth)
        dsc = torch.mean(dsc)

        return 1 - dsc

if __name__ == "__main__":
    x = torch.rand(4, 4, 224, 224)
    target = torch.randint(0, 4, [4, 224, 224]).long()
    loss_fuc = CELoss()
    loss1 = F.cross_entropy(x, target)
    loss2 = loss_fuc(x, target, torch.Tensor([1., 1., 1., 1.]))
    print(loss1, loss2)
