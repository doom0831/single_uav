import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('./ZoeDepth')
from model.actor import Actor
from model.sub_actor import SubActor
from model.critic_double import Critic
from model.replay_buffer import ReplayBuffer
from model.vae import VAE
from ZoeDepth.zoedepth.models.model_io import load_wts
from ZoeDepth.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from ZoeDepth.models.yolo import Model


pretrained_resource = "E:/single_uav_ddpg/ZoeDepth/results/0422_RGB_best.pt"
kwargs = {
            "name": "ZoeDepth",
            "version_name": "v1",
            "n_bins": 64,
            "bin_embedding_dim": 128,
            "bin_centers_type": "normed",
            "n_attractors":[4, 1],
            "attractor_alpha": 1000,
            "attractor_gamma": 2,
            "attractor_kind" : "mean",
            "attractor_type" : "inv",
            "min_temp": 0.0212,
            "max_temp": 50.0,
            "output_distribution": "logbinomial",
            "img_size":[144, 256]
}
core = Model.build(**kwargs)

class DDPG:
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.critic = Critic(cfg.n_state, cfg.n_action).to(self.device)
        self.actor = Actor(cfg.n_state, cfg.n_action).to(self.device) # 23
        self.actor_ao = SubActor(cfg.n_state_so, cfg.n_sub_action).to(self.device) # 13
        self.actor_ag = SubActor(cfg.n_state_sg, cfg.n_sub_action).to(self.device) # 2
        self.target_critic = Critic(cfg.n_state, cfg.n_action).to(self.device)
        self.target_actor = Actor(cfg.n_state, cfg.n_action).to(self.device)
        self.vae = VAE().to(self.device)
        self.vae.load('./vae.pt')
        self.zoedepth = ZoeDepth(core, **kwargs).to(self.device)
        self.zoedepth_model = load_wts(self.zoedepth, pretrained_resource)
        self.total_it = 0
        self.policy_delay = cfg.policy_delay
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip

        # 复制参数到目标网络
        # param是online网络的参数，tensor1.copy_(tensor2)，将2的元素复制给1
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=cfg.vae_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # 软更新率
        self.gamma = cfg.gamma  # 折扣率

    '''
    FloatTensor()将np.ndarray转为tensor
    unsqueeze()在指定的位置上增加1维的维度，如(2,3).unsqueeze(0)后变成(1,2,3)
    '''
    def choose_action(self, state):
        # 变成二维tensor，[1,3]，因为一维的标量不能做tensor的乘法，actor中第一层的weight形状为[3,512](标量也可以做乘法)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 處理13個感測器的動作當作ao
        if state.shape[1] == 32:
            action = self.actor_ao(state)
        # 處理終點和角度的動作當作ag
        elif state.shape[1] == 2:
            action = self.actor_ag(state)
        else:
            action = self.actor(state)
        # tensor.detach()与tensor.data的功能相同，但是若因外部修改导致梯度反向传播出错，.detach()会报错，.data不行，且.detach()得到的数据不带梯度
        action = action.detach().cpu().squeeze(0).numpy()
        return action
    
    # def vae_loss(self, recon_x, x, mu, logvar):
    #     # if x.size(1) == 3:
    #     #     x = TF.rgb_to_grayscale(x, num_output_channels=1)
    #     batch_size = x.size(0)
    #     recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    #     kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    #     return recon_loss + kl_div
    
    # def update_vae(self, x):
    #     recon_x, mu, logvar = self.vae(x)
    #     loss = self.vae_loss(recon_x, x, mu, logvar)
    #     self.vae_optimizer.zero_grad()
    #     loss.backward()
    #     self.vae_optimizer.step()

    #     return loss.item()

    def show_image(self, x):
        depth_estimation = self.zoedepth_model(x)['metric_depth']
        depth_estimation_resize = F.interpolate(depth_estimation, size=(144, 256), mode='bilinear', align_corners=False)
        depth_estimation_resize = torch.clamp(depth_estimation_resize, 0, 100) / 100.0
        recon_x, _, _ = self.vae(depth_estimation_resize)
        org_img = x[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # org_img = x[0].squeeze(0).cpu().detach().numpy()
        recon_img = recon_x[0].squeeze(0).cpu().detach().numpy()
        de_img = depth_estimation_resize.squeeze(0).squeeze(0).cpu().detach().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(org_img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(recon_img)
        ax[1].set_title('Reconstructed Image')
        ax[1].axis('off')
        ax[2].imshow(de_img)
        ax[2].set_title('Depth Estimation')
        ax[2].axis('off')
        plt.show()

    
    def update(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放缓冲池中随机采样一个批量的transition，每次用一个batch_size的数据进行训练
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # 转变为张量，得到的都是二维的tensor，第一个数字是batch_size，state(256,47)，reward(256,1)
        state = torch.FloatTensor(np.array(state)).to(self.device)  # 计算出来直接是一个二维的tensor
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(next_state) + noise).clamp(-1, 1)
            target_q1_value, target_q2_value = self.target_critic(next_state, next_action)
            target_q_value = torch.min(target_q1_value, target_q2_value)
            expected_q_value = reward + (1.0 - done) * self.gamma * target_q_value

        q1_value, q2_value = self.critic(state, action)
        smooth_loss = nn.MSELoss()
        critic_loss = smooth_loss(q1_value, expected_q_value.detach()) + smooth_loss(q2_value, expected_q_value.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            critic_input = torch.cat([state, self.actor(state)], 1)
            policy_gradient = -self.critic.q1(critic_input).mean()
            self.actor_optimizer.zero_grad()
            policy_gradient.backward()
            self.actor_optimizer.step()

            # target网络更新，软更新
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) +
                    param.data * self.soft_tau
                )
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) +
                    param.data * self.soft_tau
                )
        self.total_it += 1

    def save(self, path, filename):
        # torch.save(self.actor.state_dict(), path + 'checkpoint.pt')  # 后缀.pt和.pth没什么区别
        torch.save(self.actor.state_dict(), os.path.join(path, filename))

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'best.pt'))

    def eval_mode(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()
        self.actor_ao.eval()
        self.actor_ag.eval()
        self.vae.eval()
        self.zoedepth.eval()


    # def load_pretrain(self, path):
    #     self.actor.load_state_dict(torch.load(path + 'best.pt'))


if __name__ == '__main__':
    reward = np.array([.5])
    state = torch.FloatTensor(reward).unsqueeze(0)
    print(state)

    input = torch.FloatTensor([[.2, .3, .4], [.4, .5, .6]])
    print(input.shape)
    test1 = nn.Linear(3, 1024)
    test2 = nn.Linear(1024, 1)
    x = test1(input)
    output = test2(x)
    print(output.shape)
    print(output)
    print(output.detach())
    print(output.mean())
