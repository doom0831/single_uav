import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        缓冲区是一个队列，容量超出时去掉开始存入的transition
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)  # 替换存入的None
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量transition
        # 解压成状态，动作等，zip()表示压缩，是将每一个的第一个元素取出来，放入[]中作为第一个元素
        # zip(*)是其逆操作，是将[]中每个元素的第一个取出来，合并成第一个结果，每个结果都是一个()
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        """
        返回当前存储的量
        """
        return len(self.buffer)


if __name__ == '__main__':
    state = [[1, 2, 3], [2, 4, 3], [4, 5, 6]]
    reward = [.1, .2, .3]
    done = [True, False, False]
    batch = zip(state, reward, done)
    # print(list(batch))  # 用list()会改变batch
    s, r, d = zip(*batch)
    print(s, r, d)
    state = torch.FloatTensor(np.array(s))
    reward = torch.FloatTensor(r).unsqueeze(0)
    done = torch.FloatTensor(np.float32(d)).unsqueeze(1)
    print(state, reward, done)
