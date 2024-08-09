import numpy as np
import matplotlib.pyplot as plt

path = './outputs/UE4 and Airsim/'

ImDDPG_reward_env2 = np.load(path + 'Ours_env1_random/results/train_ma_rewards.npy')
DDPG_reward_env2 = np.load(path + '20240106-215310/results/train_ma_rewards.npy')

plt.plot(ImDDPG_reward_env2, label='ImDDPG')
plt.plot(DDPG_reward_env2, label='DDPG')
plt.title('Training Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
