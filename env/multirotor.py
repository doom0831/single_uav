import airsim
import random
import numpy as np
import torch
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import sys
import csv
import time
import threading
sys.path.append('./ZoeDepth')

from airsim import MultirotorClient
from math import *
from PIL import Image
from airsim import YawMode
from model.vae import VAE
from ZoeDepth.zoedepth.models.model_io import load_wts
from ZoeDepth.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from ZoeDepth.models.yolo import Model
matplotlib.use('TkAgg')


def normal_func(x):
    y = 1 / (1 + math.exp(-x * 2.4)) - 0.5
    # y = - (1 / (1 + math.exp(-x / 30)) - 0.5)
    return y

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

class Multirotor:
    def __init__(self, cfg, client: MultirotorClient):
        self.client = client

        start_y = random.uniform(-25, 25)
        self.steps = 1
        self.check = 1
        self.target_index = 0
        self.target_reached = False
        
        client.reset()
        client.enableApiControl(True)  # 获取控制权
        client.armDisarm(True)  # 解锁py
        # client.takeoffAsync().join()  # 起飞
        # self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, start_y, 0), airsim.to_quaternion(0, 0, 0)), True)
        client.takeoffAsync()
        client.moveToZAsync(-5, 5).join()

        self.ux, self.uy, self.uz, self.vx, self.vy, self.vz = self.get_kinematic_state()

        self.bound_x = [-110, 110]
        self.bound_y = [-50, 50]
        self.bound_z = [-20, 0]
        # self.target_x = [-350, -150]
        # self.target_y = [250, 450]
        # self.target_z = [-100, -50]
        self.target_x = [100, 100]
        self.target_y = [-15, 15]
        self.target_z = [-10, -10]
        self.d_safe = 25
        #測試目標追蹤
        # self.targets = [
        #     {"x": 100, "y": 0, "z": -10},
        #     {"x": 87.5, "y": 14.75, "z": -10},
        #     {"x": 82.25, "y": -14.75, "z": -10},
        #     {"x": 65.5, "y": 10.25, "z": -10},
        #     {"x": 62.25, "y": -10.25, "z": -10},

        # ]
        # self.targets_groups = [
        #     # [   #右1
        #     #     {"x": 100.12354, "y": 12.5645654, "z": -10},
        #     #     {"x": 112.5413231, "y": 12.5786786, "z": -10},
        #     #     {"x": 125.123123, "y": 0.1426521, "z": -10},
        #     #     {"x": 137.5123154, "y": 0.243567866, "z": -10},
        #     #     {"x": 150, "y": 0.1231546, "z": -10},
        #     # ],
        #     # [   #左1
        #     #     {"x": 100.12354, "y": -12.5645654, "z": -10},
        #     #     {"x": 112.5413231, "y": -12.5786786, "z": -10},
        #     #     {"x": 125.123123, "y": 0.1426521, "z": -10},
        #     #     {"x": 137.5123154, "y": 0.243567866, "z": -10},
        #     #     {"x": 150, "y": 0.1231546, "z": -10},
        #     # ],
        #     # [   #右2
        #     #     {"x": 100.12354, "y": 12.5645654, "z": -10},
        #     #     {"x": 112.5413231, "y": 12.5786786, "z": -10},
        #     #     {"x": 125.123123, "y": 25.1426521, "z": -10},
        #     #     {"x": 137.5123154, "y": 25.243567866, "z": -10},
        #     #     {"x": 142.5213489, "y": 25.1231546, "z": -10},
        #     #     {"x": 150, "y": 0.1231546, "z": -10},
        #     # ],
        #     # [   #左2
        #     #     {"x": 100.12354, "y": -12.5645654, "z": -10},
        #     #     {"x": 112.5413231, "y": -12.5786786, "z": -10},
        #     #     {"x": 125.123123, "y": -25.1426521, "z": -10},
        #     #     {"x": 137.5123154, "y": -25.243567866, "z": -10},
        #     #     {"x": 142.5213489, "y": -25.1231546, "z": -10},
        #     #     {"x": 150, "y": 0.1231546, "z": -10},
        #     # ]
        # ]
        # self.targets = random.choice(self.targets_groups)

        # 目标点坐标
        self.tx, self.ty, self.tz = self.generate_target()
        # self.tx, self.ty, self.tz = self.get_next_target()
        # self.tx, self.ty, self.tz = 85, 10, -10
        print("target:", self.tx, self.ty, self.tz)
        self.init_distance = self.get_distance()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vae = VAE().to(self.device)
        self.vae.load('./vae.pt')
        self.zoedepth = ZoeDepth(core, **kwargs).to(self.device)
        self.zoedepth_model = load_wts(self.zoedepth, pretrained_resource)

    # def random_target(self):
    #     target = random.choice(self.targets)
    #     return target['x'], target['y'], target['z']

    def get_next_target(self):
        if self.target_index >= len(self.targets):
            return 150, 0.102645, -10
        target = self.targets[self.target_index]
        self.target_index += 1
        return target['x'], target['y'], target['z']

    def generate_target(self):
        """
        生成目标点的位置
        seed为随机种子
        """
        tx = np.random.rand() * (self.target_x[1] - self.target_x[0]) + self.target_x[0]
        ty = np.random.rand() * (self.target_y[1] - self.target_y[0]) + self.target_y[0]
        tz = np.random.rand() * (self.target_z[1] - self.target_z[0]) + self.target_z[0]
        return tx, ty, tz
    
    def save_position(self, uy, ux, filename='drone_positions_best_env4.csv'):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([uy, ux])
              
    '''
    獲取無人機位置、速度
    '''
    def get_kinematic_state(self):
        kinematic_state = self.client.simGetGroundTruthKinematics()
        # position
        ux = float(kinematic_state.position.x_val)
        uy = float(kinematic_state.position.y_val)
        uz = float(kinematic_state.position.z_val)
        # velocity
        vx = float(kinematic_state.linear_velocity.x_val)
        vy = float(kinematic_state.linear_velocity.y_val)
        vz = float(kinematic_state.linear_velocity.z_val)
        
        # self.save_position(uy, ux)
        
        return ux, uy, uz, vx, vy, vz

    '''
    获取无人机与目标点的连线与无人机第一视角方向（飞行方向）的夹角
    tx,ty,tz为目标点坐标
    '''

    def get_deflection_angle(self):
        # 连线向量
        ax = self.tx - self.ux
        ay = self.ty - self.uy
        az = self.tz - self.uz

        # 速度方向向量
        bx = self.vx
        by = self.vy
        bz = self.vz

        if bx == 0 and by == 0 and bz == 0:  # 若无人机停止飞行，则判定完全偏航，给予一个惩罚
            return 180

        model_a = pow(ax ** 2 + ay ** 2 + az ** 2, 0.5)
        model_b = pow(bx ** 2 + by ** 2 + bz ** 2, 0.5)

        cos_ab = (ax * bx + ay * by + az * bz) / (model_a * model_b)
        radius = acos(cos_ab)  # 计算结果为弧度制，范围（0， PI），越小越好
        angle = np.rad2deg(radius)

        return angle

    def get_distance(self):
        return pow((self.tx - self.ux) ** 2 + (self.ty - self.uy) ** 2 + (self.tz - self.uz) ** 2, 0.5)

    '''
    距离传感器返回的距离数据，水平，竖直各半个圆周，每30度一采样，
    共13个数据，顺序为S、Y(1-6)、P(1-6)
    '''

    # def get_distance_sensors_data(self):
    #     yaw_axis = ['Y', 'P']
    #     pitch_axis = ['1', '2', '3', '4', '5', '6']
    #     data = []
    #     prefix = "Distance"
    #     data.append(self.client.getDistanceSensorData(distance_sensor_name=prefix + 'S').distance)
    #     for i in yaw_axis:
    #         for j in pitch_axis:
    #             dsn = prefix + i + j
    #             data.append(self.client.getDistanceSensorData(distance_sensor_name=dsn).distance)
    #     return data
    
    '''
    返回深度图数据(numpy.array)
    '''
    def extract_region_avg_depth(self, img, center, h, w):
        row_start = max(center[0] - h // 2, 0)
        row_end = min(center[0] + h // 2, img.shape[0])
        col_start = max(center[1] - w // 2, 0)
        col_end = min(center[1] + w // 2, img.shape[1])
        region = img[row_start:row_end, col_start:col_end]

        if region.size == 0:
            return 0
        
        return np.nanmean(region)
    
    def depth_image_data(self):
        data = []
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])[0]
        img_2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        img_normailized = np.clip(img_2d, 0, 100)
        
        h, w = img_normailized.shape
        region_height = h // 4
        region_width = w // 4
        
        regions = [(i * region_height + region_height // 2, j * region_width + region_width // 2)
                   for i in range(4) for j in range(4)]
        for region in regions:
            avg_depth = self.extract_region_avg_depth(img_normailized, region, region_height, region_width)
            data.append(avg_depth)
        
        return data

    def get_depth_image_data(self):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        if response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
            print(" Image data is empty, use default image")
            img_2d = np.zeros((144, 256, 3), dtype=np.uint8)
        else:
            img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_2d = img_1d.reshape(response.height, response.width, 3)
        img_normailized = img_2d.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normailized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])[0]
        # img_2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        
        # if img_2d.shape[0] != 144 or img_2d.shape[1] != 256:
        #     print(" Image data shape is not valid, use default image")
        #     img_2d = np.zeros((144, 256), dtype=np.float32)
        
        # img_2d = np.clip(img_2d, 0, 100) / 100.0
        
        # print(' img_2d: ', img_2d.shape)
        # img_tensor = torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        # plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        # plt.show()
        

        # def draw_region(center, height, width, color='r'):
        #     rect = patches.Rectangle((center[1] - width // 2, center[0] - height // 2), width, height, linewidth=1, edgecolor=color, facecolor='none')
        #     plt.gca().add_patch(rect)
        #     # draw_region(region, region_height, region_width)
  

        # plt.show()
        return img_tensor
    
    '''
    返回无人机状态(numpy.array)
    '''
    def get_state(self):
        # 进行归一化
        position = np.array([self.tx - self.ux, self.ty - self.uy, self.tz - self.uz])
        target = np.array([self.get_distance() / self.init_distance])
        velocity = np.array([self.vx, self.vy, self.vz])
        angle = np.array([self.get_deflection_angle() / 180])
        # sensor_data = np.array(self.get_distance_sensors_data()) / 20
        # depth_image_data = np.array(self.depth_image_data()) / 50
        img_tensor = self.get_depth_image_data()
        
        with torch.no_grad():
            depth_estimation = self.zoedepth_model(img_tensor)['metric_depth']
            depth_estimation_resize = F.interpolate(depth_estimation, size=(144, 256), mode='bilinear', align_corners=False)
            depth_estimation_resize = torch.clamp(depth_estimation_resize, 0, 100) / 100.0
            # plt.imshow(depth_estimation_resize.squeeze(0).squeeze(0).cpu().numpy())
            # plt.colorbar()
            # plt.show()
            latent_code = self.vae(depth_estimation_resize, return_latent_code=True)
        # latent_code = self.vae(img_tensor, return_latent_code=True)

        latent_code_np = latent_code.cpu().detach().numpy().flatten()

        state = np.append(position, target)
        state = np.append(state, velocity)
        state = np.append(state, angle)
        # state = np.append(state, sensor_data)
        # state = np.append(state, depth_image_data)
        state = np.append(state, latent_code_np)

        return state

    '''
    计算当前状态下的奖励、是否完成
    加速度的坐标为NED，z轴加速度为负则向上飞行
    三个加速度有统一的范围
    '''
    # 碰撞惩罚要略大于目标点惩罚
    def step(self, action):
        # start_time = time.time()
        done = self.if_done()
        #更新目標點
        # if self.steps % 10 == 0:
        #     self.tx, self.ty, self.tz = self.random_target()
        #     print(" \ntarget:", self.tx, self.ty, self.tz)
        # self.steps += 1

        # 抵達目標區域開始追蹤
        # if not self.target_reached and self.get_distance() <= 30:
        #     self.tx, self.ty, self.tz = self.get_next_target()
        #     self.check = 6
        #     self.target_reached = True
        #     print(" \ntarget:", self.tx, self.ty, self.tz)

        # if self.target_reached and self.steps % self.check == 0:
        #     self.tx, self.ty, self.tz = self.get_next_target()
        #     print(" \ntarget:", self.tx, self.ty, self.tz)
        # if self.target_reached:
        #     self.steps += 1

        arrive_reward = self.arrive_reward()
        yaw_reward = self.yaw_reward()
        # min_sensor_reward = self.min_sensor_reward()
        # num_sensor_reward = self.num_sensor_reward()
        collision_reward = 0
        step_reward = self.step_reward()
        z_distance_reward = self.z_distance_reward()
        cross_border_reward = self.cross_border_reward()
        depth_image_reward = self.depth_image_reward()
        collision_times = 0


        # 碰撞完成与碰撞奖励一起做
        if self.client.simGetCollisionInfo().has_collided:
            collision_times = 1
            # 发生碰撞
            done = True
            '''
            碰撞惩罚-8
            '''
            collision_reward = -8

        ax = action[0]
        ay = action[1]
        az = action[2]
        my_yaw_mode = YawMode(False, 0)
        # TODO 尝试改为速度控制
        self.client.moveByVelocityAsync(vx=self.vx + ax,
                                        vy=self.vy + ay,
                                        vz=self.vz + az,
                                        duration=0.5,
                                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                                        yaw_mode=my_yaw_mode)

        kinematic_state = self.client.simGetGroundTruthKinematics()
        distance_reward = self.distance_reward(kinematic_state.position.x_val, kinematic_state.position.y_val, kinematic_state.position.z_val)
        # distance_reward = self.distance_reward(self.ux, self.uy, self.uz)
        # reward = arrive_reward + yaw_reward  + min_sensor_reward + collision_reward + step_reward + distance_reward + cross_border_reward + z_distance_reward
        reward = arrive_reward + yaw_reward + collision_reward + step_reward + distance_reward + cross_border_reward + z_distance_reward + depth_image_reward
        
        self.ux, self.uy, self.uz, self.vx, self.vy, self.vz = self.get_kinematic_state()
        next_state = self.get_state()
        time.sleep(0.15)

        # end_time = time.time()
        # processing_time = end_time - start_time
        # print(f" Processing time: {processing_time:.4f}s")
        
        return next_state, reward, done, collision_times

    def if_done(self):
        # 与目标点距离小于25米
        model_a = self.get_distance()
        if model_a <= 10.0:
            return True
        # 触及边界
        if self.ux < self.bound_x[0] or self.ux > self.bound_x[1] or \
                self.uy < self.bound_y[0] or self.uy > self.bound_y[1] or \
                self.uz < self.bound_z[0]:
            return True
        # # 超过目標最大距离
        # if self.get_distance() > 105.0:
        #     return True

        return False

    '''
    越界惩罚-8
    '''

    def cross_border_reward(self):
        if self.ux < self.bound_x[0] or self.ux > self.bound_x[1] or \
                self.uy < self.bound_y[0] or self.uy > self.bound_y[1] or \
                self.uz < self.bound_z[0]:
            return -8
        return 0

    '''
    与目标点距离惩罚(-0.5,0.5)，两种方式需要权衡，另一种是（-.5,0）,func=-(1 / (1 + math.exp(-x / 30)) - 0.5)
    '''

    def distance_reward(self, next_ux, next_uy, next_uz):
        model_a = self.get_distance()
        xa = self.tx - next_ux
        ya = self.ty - next_uy
        za = self.tz - next_uz
        model_b = pow(xa ** 2 + ya ** 2 + za ** 2, 0.5)
        return normal_func(model_a-model_b)
    
    '''
    高度懲罰    
    '''

    def z_distance_reward(self):
        reward = 0
        z_distance = abs(self.tz - self.uz)
        if z_distance > 15:
            reward += -0.2
        elif 15 >= z_distance > 12.5:
            reward += -0.15
        else:
            reward += 0.1
        return reward

    '''
    抵达目标点奖励+12
    '''

    def arrive_reward(self):
        x = self.tx - self.ux
        y = self.ty - self.uy
        z = self.tz - self.uz
        model_a = pow(x ** 2 + y ** 2 + z ** 2, 0.5)
        if model_a <= 10.0:
            return 12
        else:
            return 0

    '''
    偏航惩罚(-0.4,0)，均值0.2
    '''

    def yaw_reward(self):
        yaw = self.get_deflection_angle()
        return -0.4 * (yaw / 180)

    '''
    最短激光雷达长度惩罚(-.8,0) 
    '''

    # def min_sensor_reward(self):
    #     sensor_data = self.get_distance_sensors_data()
    #     d_min = min(sensor_data)
    #     if d_min < self.d_safe:
    #         # return -0.1 * (self.d_safe - d_min)
    #         # return 0.9 * (math.exp((self.d_safe - d_min) / -6) - 1)
    #         return 0.5 * (math.exp((self.d_safe - d_min) / -5) - 1)
    #     else:
    #         return 0
    
    '''
    深度圖惩罰
    '''
    def depth_image_reward(self):
        depth_image_distance = self.depth_image_data()
        d_min = min(depth_image_distance)
        if d_min < self.d_safe:
            return 0.5 * (math.exp((self.d_safe - d_min) / -5) - 1)
            # return -max(0, (self.d_safe - d_min) / self.d_safe)
            # return -0.05 * (self.d_safe - d_min)
        else:
            return 0
        # img_tensor = self.get_depth_image_data()
        # depth_estimation = self.zoedepth_model(img_tensor)['metric_depth']
        # depth_estimation_resize = F.interpolate(depth_estimation, size=(144, 256), mode='bilinear', align_corners=False)
        # depth_estimation_resize = torch.clamp(depth_estimation_resize, 0, 100) / 100.0
        # depth_map, _, _ = self.vae(depth_estimation_resize)
        # depth_map = depth_map.detach().cpu().numpy().squeeze()
        
        # N = depth_map.mean()
        # H, W = depth_map.shape[-2:]
        # center_region = depth_map[H//2-32:H//2+32, W//2-32:W//2+32]
        # M = center_region.mean()
        # if M > N:
        #     reward = 0
        # else:
        #     reward = -0.2
        # return reward


    '''
    小于安全阈值的激光雷达条数惩罚(-0.4,0), 均值0.2
    '''

    # def num_sensor_reward(self):
    #     sensor_data = self.get_distance_sensors_data()
    #     num = sum(i < self.d_safe for i in sensor_data)
    #     return -0.4 * (num / len(sensor_data))

    '''
    漫游惩罚-0.02
    '''

    def step_reward(self):
        if not self.if_done():
            # return -0.02
            return -0.2
        else:
            return 0

    # def save_image(self, image_data, folder, idx, cmap=None):
    #     if cmap is None:
    #         img = Image.fromarray(image_data)
    #         img.save(os.path.join(folder, f'image_{idx}.png'))
    #     else:
    #         plt.imsave(os.path.join(folder, f'image_{idx}.png'), image_data, cmap=cmap)
    
    # def get_image_data(self, save_dir, image_idx=0):
    #     # os.makedirs(os.path.join(save_dir, 'RGB'), exist_ok=True)
    #     # os.makedirs(os.path.join(save_dir, 'Depth'), exist_ok=True)
    #     # os.makedirs(os.path.join(save_dir, 'DepthValues'), exist_ok=True)
    #     # os.makedirs(os.path.join(save_dir, 'CameraInfo'), exist_ok=True)
    #     os.makedirs(os.path.join(save_dir, 'DE'), exist_ok=True)

    #     # response_rgb = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    #     # response_depth = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])[0]
    #     # if response_rgb.image_data_uint8 is None or len(response_rgb.image_data_uint8) == 0:
    #     #     print(" Image data is empty, use default image")
    #     #     img_rgb = np.zeros((144, 256, 3), dtype=np.uint8)
    #     # else:
    #     #     img_1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
    #     #     img_rgb = img_1d.reshape(response_rgb.height, response_rgb.width, 3)
        
    #     # img_depth = airsim.list_to_2d_float_array(response_depth.image_data_float, response_depth.width, response_depth.height)
    #     # if img_depth.shape[0] != 144 or img_depth.shape[1] != 256:
    #     #     print(" Image data shape is not valid, use default image")
    #     #     img_depth = np.zeros((144, 256), dtype=np.float32)
    #     # img_depth = np.clip(img_depth, 0, 100)
        
    #     # #camera_info
    #     # camera_position = response_depth.camera_position
    #     # camera_orientation = response_depth.camera_orientation

    #     # camera_info = {
    #     #     "position": {
    #     #         "x": camera_position.x_val,
    #     #         "y": camera_position.y_val,
    #     #         "z": camera_position.z_val
    #     #     },
    #     #     "orientation": {
    #     #         "w": camera_orientation.w_val,
    #     #         "x": camera_orientation.x_val,
    #     #         "y": camera_orientation.y_val,
    #     #         "z": camera_orientation.z_val
    #     #     }
    #     # }

    #     # camera_info_path = os.path.join(save_dir, 'CameraInfo', f'image_{image_idx}.txt')
    #     # with open(camera_info_path, 'w') as f:
    #     #     f.write(f"{camera_info['position']['x']} ")
    #     #     f.write(f"{camera_info['position']['y']} ")
    #     #     f.write(f"{camera_info['position']['z']} ")
    #     #     f.write(f"{camera_info['orientation']['w']} ")
    #     #     f.write(f"{camera_info['orientation']['x']} ")
    #     #     f.write(f"{camera_info['orientation']['y']} ")
    #     #     f.write(f"{camera_info['orientation']['z']}")

    #     img_tensor = self.get_depth_image_data()
    #     depth_estimation = self.zoedepth_model(img_tensor)['metric_depth']
    #     depth_estimation_resize = F.interpolate(depth_estimation, size=(144, 256), mode='bilinear', align_corners=False)
    #     depth_estimation_resize = torch.clamp(depth_estimation_resize, 0, 100) / 100.0
    #     img_de = depth_estimation_resize.squeeze(0).squeeze(0).cpu().detach().numpy()

    #     # self.save_image(img_rgb, os.path.join(save_dir, 'RGB'), image_idx)
    #     # self.save_image(img_depth, os.path.join(save_dir, 'Depth'), image_idx, cmap='viridis')
    #     self.save_image(img_de, os.path.join(save_dir, 'DE'), image_idx, cmap='viridis')

    #     # np.savetxt(os.path.join(save_dir, 'DepthValues', f'image_{image_idx}.txt'), img_depth, fmt='%.2f')

if __name__ == '__main__':
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    mr = Multirotor(client)
    data = np.array(mr.get_distance_sensors_data()) / mr.d_safe
    print(data)
