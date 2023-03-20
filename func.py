#%%
from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
#PyTorch
class SSLloss(nn.Module):
    def __init__(self):
        super(SSLloss, self).__init__()
    def forward(self, prediction, actual,  c):
        loss = (torch.sigmoid((prediction - 20) * c) - torch.sigmoid((actual - 20) * c)) ** 2
        loss = loss.sum()
        return loss
class Weightloss(nn.Module):
    def __init__(self):
        super(Weightloss, self).__init__()
    def forward(self, prediction, actual, thresholds, weight_interval):
        weights = torch.ones_like(actual) * weight_interval[0]
        for i, threshold in enumerate(thresholds):
            weights = weights + (weight_interval[i + 1] - weight_interval[i]) * (actual >= threshold)
        loss = (torch.pow((prediction - actual), 2) * weights).mean()
        return loss

class BMSELoss(torch.nn.Module):
    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 20, 50, 100, 300]
        self.y_l = [1/20, 10/20, 20/20, 50/20, 100/20]
    def forward(self, x, y): # x is prediction / y is actual
        w = y.clone()  # w = weight
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]  # w<0.283, will be give t he weight of 1
        return torch.mean(w * ((y - x)** 2))

class TSLoss(nn.Module):
    def __init__(self):
        super(TSLoss, self).__init__()
	# target contain 0 and 1 only.
	# input is - -> +
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

#PyTorch
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

#PyTorch
class BIASLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BIASLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1).sum() 
        targets = targets.view(-1).sum() 
        
        bias = inputs / targets
        return torch.abs(bias - smooth)
    

#PyTorch  用于one hot输出层的IoU
class MulTsLoss(nn.Module):
    def __init__(self):
        super(MulTsLoss, self).__init__()
    def forward(self, inputs, targets, classnum = 3):
        ts_sum = 0
        ts_list = []
        for i in range(1,classnum):
            inputs_cal_ts = torch.max(inputs[:,i:],dim = 1)[0].flatten()
            targets_cal_ts = torch.max(targets[:,i:],dim = 1)[0].flatten()
            intersection = (inputs_cal_ts * targets_cal_ts).sum()        
            total = (inputs_cal_ts + targets_cal_ts).sum()
            union = total - intersection  
            ts_sum += intersection / union
            ts_list.append(intersection / union)
        ts_mean = ts_sum / (classnum - 1)
        print(ts_list)
        return 1 - ts_mean

#PyTorch
class MulTsRegLoss(nn.Module):
    def __init__(self):
        super(MulTsRegLoss, self).__init__()
    def forward(self, inputs, targets, thresholds = [35/70]):
        
        ts_sum = 0
        
        inputs = torch.clamp(inputs, 0, 1)
        
        targets = torch.clamp(targets, 0, 1)
        
        for i in range(len(thresholds)):
            
            inputs_copy = inputs.flatten().clone()
            
            # inputs_copy[inputs.flatten() >= thresholds[i]] = 1
            
            targets_copy = targets.flatten().clone()
            
            targets_copy[targets.flatten() < thresholds[i]] = 0
            
            targets_copy[targets.flatten() >= thresholds[i]] = 1
            
            intersection = (inputs_copy.flatten() * targets_copy).sum()
            
            total = (inputs_copy.flatten() + targets_copy).sum()
            
            union = total - intersection
            
            ts_sum += intersection / union
            
        ts_mean = ts_sum / (len(thresholds))
        
        return 1 - ts_mean
#PyTorch
class MulBIASRegLoss(nn.Module):
    def __init__(self):
        super(MulBIASRegLoss, self).__init__()
    def forward(self, inputs, targets, thresholds = [35/70]):
        
        bias_sum = 0
        
        for i in range(len(thresholds)):
            
            inputs_copy = inputs.flatten().clone()
            
            inputs_copy[inputs.flatten() >= thresholds[i]] = 1 
            
            targets_copy = targets.flatten().clone()

            targets_copy[targets.flatten() < thresholds[i]] = 0
            
            targets_copy[targets.flatten() >= thresholds[i]] = 1
            
            bias = inputs_copy.sum() / targets_copy.sum()
            
            bias_sum += bias
            
        bias_mean = bias_sum / (len(thresholds))
        return torch.abs(bias_mean - 1) * 0.1




