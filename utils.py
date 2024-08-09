#!/usr/bin/env python
# coding=utf-8
"""
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2022-07-13 22:15:46
Discription:
Environment:
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties  # 导入字体模块


def chinese_font():
    """
    设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    """
    try:
        font = FontProperties(
            fname='/System/Library/Fonts/STHeiti Light.ttc', size=15)  # fname系统字体路径，此处是mac的
    except:
        font = None
    return font


def plot_rewards_cn(rewards, ma_rewards, cfg, tag='train'):
    """
    中文画图
    """
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg.env_name,
                                       cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg.save:
        plt.savefig(cfg.result_path + f"{tag}_rewards_curve_cn")
    # plt.show()


def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve".format(tag))
    plt.show()

def plot_rewards_test(rewards, ma_rewards, cfg, tag='test'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve".format(tag))
    plt.show()


def plot_losses(losses, cfg, tag='train'):
    sns.set()
    plt.figure()
    plt.title("loss curve of VAE")
    plt.xlabel('steps')
    plt.ylabel('losses')
    plt.plot(losses, label='train losses')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_losses_curve".format(tag))
    plt.show()


def save_results_1(dic, tag='train', path='./results'):
    """
    保存奖励
    """
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    print('Results saved！')


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """
    保存奖励
    """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    # np.save(path + '{}_images.npy'.format(tag), images)
    print('Result saved!')

def save_results_test(rewards, ma_rewards, tag='test', path='./results'):
    """
    保存奖励
    """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def make_dir(*paths):
    """
    创建文件夹
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """
    删除目录下所有空文件夹
    """
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_args(args, success_rate, total_time, collision_rate, outside_rate):
    # save parameters    
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('total_time : {:.2f}hr'.format(total_time) + '\n')
        f.writelines('success_rate : {:.3f}'.format(success_rate)  + '\n')
        f.writelines('collision_rate : {:.3f}'.format(collision_rate)  + '\n')
        f.writelines('outside_rate : {:.3f}'.format(outside_rate)  + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")

def save_args_test(args, success_rate, collision_rate, outside_rate, avg_step):
    # save parameters    
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('success_rate : {:.4f}'.format(success_rate)  + '\n')
        f.writelines('collision_rate : {:.4f}'.format(collision_rate)  + '\n')
        f.writelines('outside_rate : {:.4f}'.format(outside_rate)  + '\n')
        f.writelines('avg_step : {:.3f}'.format(avg_step)  + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")
