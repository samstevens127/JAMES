import torch
import torch.nn as nn
import torch.nn.functional as F

class ShogiNet(nn.Module):
    def __init__(self):
        super(ShogiNet, self).__init__()
        self.conv1 = nn.Conv2d(48, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 9 * 9, 13932)
        
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        p = F.relu(self.policy_conv(x))
        p = self.policy_fc(p.view(-1, 2 * 9 * 9))
        
        v = F.relu(self.value_conv(x))
        v = F.relu(self.value_fc1(v.view(-1, 1 * 9 * 9)))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
