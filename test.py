import airsim
import numpy as np
import time
import torch

from torch.utils.tensorboard import SummaryWriter
# from model.ddpg import DDPG
from model.ddpg_double import DDPG
from env.multirotor import Multirotor
from train import get_args, set_seed
from utils import save_results_test, plot_rewards_test, make_dir, save_args_test


def test(cfg, client, agent):
    print('Start testing')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    writer = SummaryWriter('./test_image')
    image_idx = 0
    success = 0
    total_collision = 0
    total_step = 0
    total_testing_time = time.time()
    for i_ep in range(cfg.test_eps):
        env = Multirotor(cfg, client)
        state = env.get_state()
        ep_reward = 0
        finish_step = 0
        final_distance = state[3] * env.init_distance
        for i_step in range(cfg.max_step):
            finish_step = finish_step + 1
            action_ao = agent.choose_action(state[8:])
            # action_ao = torch.from_numpy(action_ao)
            action_ag = agent.choose_action(np.array([state[3], state[7]]))
            # action_ag = agent.choose_action(state[[3, 7]])
            # action_ag = torch.from_numpy(action_ag)
            combine_state = np.concatenate([state, action_ao, action_ag])
            # combine_state = torch.cat([state, action_ao, action_ag], dim=0)
            # action = agent.choose_action(state)
            action = agent.choose_action(combine_state)
            next_state, reward, done, collision = env.step(action)
            ep_reward += reward
            state = next_state

            # env.get_image_data(cfg.result_path, image_idx)
            # image_idx += 1

            # img_tensor = env.get_depth_image_data()
            # agent.show_image(img_tensor)

            print('\rEpisode: {} Step: {} Reward: {:.2f} Distance: {}'.format(i_ep + 1, i_step + 1, ep_reward,
                                                                                 state[3] * env.init_distance), end="")
            final_distance = state[3] * env.init_distance
            if done:
                break
        print('\rEpisode: {} Finish step: {} Reward: {:.2f} Final distance: {}'.format(i_ep + 1, finish_step,
                                                                                          ep_reward, final_distance))
        if final_distance <= 10.0:
            success += 1
        if collision:
            total_collision += 1
        else:
            total_step += finish_step
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        writer.add_scalars(main_tag='test',
                           tag_scalar_dict={
                               'reward': ep_reward,
                               'ma_reward': ma_rewards[-1]
                           },
                           global_step=i_ep)
        print(f'Added scalar data for episode {i_ep+1} - Reward: {ep_reward}, MA Reward: {ma_rewards[-1]}')
    
    avg_step = total_step / (cfg.test_eps - total_collision)
    success_rate = success / cfg.test_eps
    collision_rate = total_collision / cfg.test_eps
    outside_rate = 1 - success_rate - collision_rate
    total_time = (time.time() - total_testing_time) / 3600
    print('Finish testing!')
    print('Total time: {:.2f}hr'.format(total_time))
    print('Average Reward: {} Success Rate: {}'.format(np.mean(rewards), success / cfg.test_eps))
    print('Collision Rate: {}'.format(total_collision / cfg.test_eps))
    print('Outside Rate: {}'.format(outside_rate))
    print('Average Step: {}'.format(avg_step))
    writer.close()

    return rewards, ma_rewards, success_rate, collision_rate, outside_rate, avg_step


# 0.63 原目标点
# 0.43 相反的固定区域目标点
# 0.58 全地图
if __name__ == '__main__':
    cfg = get_args()
    set_seed(cfg.seed)
    make_dir(cfg.result_path)
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    agent = DDPG(cfg=cfg)
    agent.load(path=cfg.model_path)
    agent.eval_mode()
    rewards, ma_rewards, success_rate, collision_rate, outside_rate, avg_step = test(cfg, client, agent)
    save_args_test(cfg, success_rate, collision_rate, outside_rate, avg_step)
    save_results_test(rewards, ma_rewards, tag='test', path=cfg.result_path)
    plot_rewards_test(rewards, ma_rewards, cfg, tag="test")  # 画出结果
