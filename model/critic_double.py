import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(Critic, self).__init__()
        #Q1 Network
        self.q1 = nn.Sequential(
            nn.Linear(n_states + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.init_weights(self.q1[-1], init_w)
        
        #Q2 Network
        self.q2 = nn.Sequential(
            nn.Linear(n_states + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.init_weights(self.q2[-1], init_w)
    
    def init_weights(self, layer, init_w):
        nn.init.uniform_(layer.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(layer.bias.detach(), a=-init_w, b=init_w)

    def forward(self, state, action):
        # 按维数1拼接(按维数1拼接为横着拼，按维数0拼接为竖着拼)
        x = torch.cat([state, action], 1)
        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


if __name__ == '__main__':
    critic = Critic(n_states=3 + 1 + 3 + 1 + 13, n_actions=3)
    print(sum(p.numel() for p in critic.parameters() if p.requires_grad))
