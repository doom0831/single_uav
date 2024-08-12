import os
import sys

import datetime
import time

import numpy as np
import random
import torch
import argparse
import airsim

from torch.utils.tensorboard import SummaryWriter
from model.ou_noise import OrnsteinUhlenbeckActionNoise as OUNoise
# from model.ddpg import DDPG
from model.ddpg_double import DDPG
from model.vae import VAE
from env.multirotor import Multirotor
from utils import save_results, make_dir, plot_rewards, save_args, plot_losses

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path


def get_args():
    """
    Hyper parameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    # curr_time = "20220728-131136"
    curr_time = "20240519-020649_best"
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='UE4 and Airsim', type=str, help="name of environment")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--n_state', default=3 + 1 + 3 + 1 + 32 + 2, type=int, help="numbers of state space")
    parser.add_argument('--n_state_so', default=32, type=int, help="numbers of state space")
    parser.add_argument('--n_state_sg', default=2, type=int, help="numbers of state space")
    parser.add_argument('--n_action', default=3, type=int, help="numbers of state action")
    parser.add_argument('--n_sub_action', default=1, type=int, help="numbers of state action")
    parser.add_argument('--update_times', default=1, type=int, help="update times")
    parser.add_argument('--train_eps', default=1200, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=100, type=int, help="episodes of testing")
    parser.add_argument('--max_step', default=500, type=int, help="max step for getting target")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--vae_lr', default=1e-4, type=float, help="learning rate of vae")
    parser.add_argument('--memory_capacity', default=2**17, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + "DDPG_test" + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # check GPU
    return args


def set_seed(seed):
    """
    全局生效
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(cfg, client, agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    ou_noise = OUNoise(mu=np.zeros(cfg.n_action))  # noise of action
    rewards, ma_rewards, loss = [], [], []
    image_buffer = []
    image_idx = 0
    count = 0
    best_reward = -float('inf')
    best_single_reward = -float('inf')
    best_single_state = None
    cumulative_reward = 0
    writer = SummaryWriter('./train_image')
    success = 0
    total_collision = 0
    total_training_time = time.time()
    total_step = 0
    for i_ep in range(cfg.train_eps):
        env = Multirotor(cfg, client)
        state = env.get_state()
        ou_noise.reset()
        ep_reward = 0
        finish_step = 0
        final_distance = state[3] * env.init_distance
        episode_start_time = time.time()
        for i_step in range(cfg.max_step):
            finish_step = finish_step + 1
            # 分別處理ao和ag的動作
            action_ao = agent.choose_action(state[8:])
            action_ag = agent.choose_action(np.array([state[3], state[7]]))
            # 把ao和ag以及總狀態資訊做合併
            combine_state = np.concatenate([state, action_ao, action_ag])
            action = agent.choose_action(combine_state)
            # action = agent.choose_action(state)
            action = action + ou_noise(i_step)  # 动作加噪声 OU噪声最大不会超过0.3
            # noise = np.random.normal(0, 0.1, size=action.shape)
            # action = action + noise
            # 加速度范围 ax[-1,1] ay[-1,1] az[-1,1] 速度大致范围 vx[-11,11] vy[-11,11] vz[-8,8]
            action = np.clip(action, -1, 1)  # 裁剪
            next_state, reward, done, collision = env.step(action)
            action_ao_next = agent.choose_action(next_state[8:])
            action_ag_next = agent.choose_action(np.array([next_state[3], next_state[7]]))
            combine_state_next = np.concatenate([next_state, action_ao_next, action_ag_next])
            ep_reward += reward
            # agent.memory.push(state, action, reward, next_state, done)
            agent.memory.push(combine_state, action, reward, combine_state_next, done)

            # img_tensor = env.get_depth_image_data()
            # image_buffer.append(img_tensor)
            # should_update = len(image_buffer) >= 32
            # should_display = (i_ep + 1) % 2000 == 0
            
            # if should_display:
            #     if count < 5:
            #         agent.show_image(img_tensor)
            #         count += 1
            # if should_update:
            #     image_buffer_tensor = torch.cat(image_buffer, dim=0).to(cfg.device)
            #     vae_loss = agent.update_vae(image_buffer_tensor)
            #     loss.append(vae_loss)
            #     print(f' VAE Loss: {vae_loss}')
            #     image_buffer = []
            #     vae_model.save(path=cfg.model_path, filename='vae_ck.pt')

            # env.get_image_data(cfg.result_path, image_idx)
            # image_idx += 1

            replay_len = len(agent.memory)
            k = 1 + replay_len / cfg.memory_capacity
            update_times = int(k * cfg.update_times)
            for _ in range(update_times):
                agent.update()

            state = next_state
            elapsed_time = time.time() - episode_start_time
            print('\rEpisode: {} Step: {} Reward: {:.2f} Distance: {:.2f} Time: {:.2f}'.format(i_ep+1, i_step+1,  ep_reward, state[3] * env.init_distance, elapsed_time), end="")
            final_distance = state[3] * env.init_distance
            if done:
                break
        
        cumulative_reward += ep_reward
        total_step += finish_step
        # ε = max(ε - 0.002, 0.001)
        # ou_noise.decay_epsilon()

        print('\rEpisode: {} Finish step: {} Reward: {:.2f} Final distance: {:.2f} Total time: {:.2f}'.format(i_ep+1, finish_step, ep_reward, final_distance, elapsed_time))
        if final_distance <= 10.0:
            success += 1
        if collision:
            total_collision += 1
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        writer.add_scalars(main_tag='train',
                           tag_scalar_dict={
                               'reward': ep_reward,
                               'ma_reward': ma_rewards[-1]
                           },
                           global_step=i_ep)
        print(f'Added scalar data for episode {i_ep+1} - Reward: {ep_reward}, MA Reward: {ma_rewards[-1]}')
        
        if ep_reward > best_single_reward:
            best_single_reward = ep_reward
            best_single_state = agent.actor.state_dict()
        if (i_ep + 1) % 50 == 0:
            avg_reward = cumulative_reward / 50
            cumulative_reward = 0
            agent.save(path=cfg.model_path, filename='last.pt')
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.actor.load_state_dict(best_single_state)
                agent.save(path=cfg.model_path, filename='best.pt')
            best_single_reward = -float('inf')
            best_single_state = None
            
    writer.close()
    success_rate = success / cfg.train_eps
    collision_rate = total_collision / cfg.train_eps
    outside_rate = 1 - success_rate - collision_rate
    total_time = (time.time() - total_training_time) / 3600
    print('Finish training!')
    #Count total training time、success rate and collision rate
    print('Total training time: {:.2f}hr Total Steps: {}'.format(total_time, total_step))
    print('Suceess rate: {}'.format(success / cfg.train_eps))
    print('Collision rate: {}'.format(total_collision / cfg.train_eps))
    print('Outside rate: {}'.format(outside_rate))
    return rewards, ma_rewards, success_rate, total_time, collision_rate, outside_rate, loss


if __name__ == '__main__':
    cfg = get_args()
    set_seed(cfg.seed)
    make_dir(cfg.result_path, cfg.model_path)
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    agent = DDPG(cfg)
    vae_model = VAE()
    rewards, ma_rewards, success_rate, total_time, collision_rate, outside_rate, loss = train(cfg, client, agent)
    save_args(cfg, success_rate, total_time, collision_rate, outside_rate)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
    plot_losses(loss, cfg, tag="train")  # 画出结果
