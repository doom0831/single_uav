import torch
import torch.nn as nn
import torch.nn.functional as F

'''
FC层默认采用kaiming(He initial)初始化的方式，对于使用relu作为激活函数的FC层会有良好的效果
torch.nn.init中包括很多初始化方法
'''


class SubActor(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(SubActor, self).__init__()
        self.linear1 = nn.Linear(n_states, 32)
        # self.dropout1 = nn.Dropout(0.2)
        # self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        # self.dropout2 = nn.Dropout(0.2)
        # self.linear4 = nn.Linear(16, 16)
        self.linear5 = nn.Linear(16, n_actions)

        # tensor.uniform_()函数，从参数的均匀分布中采样进行填充
        nn.init.uniform_(self.linear5.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.linear5.bias.detach(), a=-init_w, b=init_w)

        # 另一种写法
        # self.linear5.weight.data.uniform_(-init_w, init_w)
        # self.linear5.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = self.dropout1(x)
        # x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = self.dropout2(x)
        # x = F.relu(self.linear4(x))
        action = torch.tanh(self.linear5(x))  # torch.tanh与F.tanh没有区别
        return action


if __name__ == '__main__':
    actor = SubActor(n_states=3 + 1 + 3 + 1 + 13, n_actions=3)
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
