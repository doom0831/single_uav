import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

path = './outputs/UE4 and Airsim/'

ddpg = np.load(path + '20240602-234920-ddpg/results/train_ma_rewards.npy')
td3 = np.load(path + '20240612-011105-td3/results/train_ma_rewards.npy')
sddpg = np.load(path + '20240605-120512-sddpg/results/train_ma_rewards.npy')
sddpgex = np.load(path + '20240531-111407-sddpg/results/train_ma_rewards.npy')
Ours = np.load(path + '20240519-020649_best/results/train_ma_rewards.npy')

window_size = 20

ddpg_smooth = moving_average(ddpg, window_size)
td3_smooth = moving_average(td3, window_size)
sddpg_smooth = moving_average(sddpgex, window_size)
Ours_smooth = moving_average(Ours, window_size)

# plt.plot(ddpg, label='DDPG')
plt.plot(ddpg_smooth, label='DDPG')
plt.plot(td3_smooth, label='TD3')
plt.plot(sddpg_smooth, label='SDDPG')
plt.plot(Ours_smooth, label='Ours')
# plt.title('Training Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.show()