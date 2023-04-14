# -*- coding: utf-8 -*-
import numpy as np
import random
import gym
import copy
from gym import spaces
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time


# 模拟串联队列环境的类
class TandemEnv(gym.Env):

    # 每个子队列的服务器的数量， 每个子队列的服务器的计算速度， 计算任务的到达速率， 延迟上界， 每个时间步骤的长度， 随机种子数， 权衡资源分配和任务延迟的参数
    def __init__(self, N_s, Mu_s, lambda_rate, d_ub, step_len,seed,tradeoff_lambda):
        # super().__init__()
        """
        N_s = array of number of servers at each stage
        Mu_s = array of service rates
        lambda_rate = arrival rate
        d_ub = delay upperbound
        step_len = length of each time slot (step)
        """
        self.epsilon_ub = 0.1
        self.seed = random.seed(seed)
        self.tradeoff_lambda = tradeoff_lambda # {8,10,12,14,16}
        self.N_s = np.array(N_s)
        # 此变量只影响 service time generation 函数
        self.Mu_s = Mu_s
        # 下面并没有用到 rho 变量
        # self.rho = rho
        self.d_ub = d_ub
        self.lambda_rate = lambda_rate

        # 唯一标定一个任务
        self.job_index = 1
        # 任务到达时间点
        self.t_arr = 0
        # 记录step数值
        self.cnt = 1
        self.priority = 10
        self.finish_onestep = False
        self.cost = 0
        self.T = step_len
        # 一个 episode 总时间长度
        # terminate episode if end2end delay exceeds the delay_max or
        # the number of steps reaches max_steps
        episode_len = 2000.0
        self.MAX_STEPS = episode_len / self.T
        self.delay_max = 40  # 80.0
        self.dep_vec = np.zeros((1, 3))
        # self.dep_vec = np.zeros((1,4))

        self.mu_min = 1.1 * self.lambda_rate / self.N_s
        self.mu_max = 2
        # 定义动作空间，和取值范围，值类型
        self.action_space = spaces.Box(
            low=self.mu_min,
            high=self.mu_max, shape=(len(N_s),),
            dtype=np.float32
        )

        self.B_max = 10
        # 生成一个向量或矩阵，里面的元素要么是1要么是0
        # self.observation_space = gym.spaces.MultiBinary(n=self.B_max * len(N_s))
        self.observation_space = np.array([spaces.Discrete(1025)]*len(N_s))

        # all queue length
        self.qls = np.zeros(len(N_s), dtype=int)
        # all queue length average
        self.qls_ave = np.zeros(len(N_s), dtype=float)
        self.tandem = Tandem(N_s, Mu_s)
        self.tandem_job_dict = {}
        self.t_slot = 0
        self.arr_num_avg = 0

    # 模拟串联队列采取动作后运行情况
    def step(self, action):
        for n_s in range(len(self.N_s)):
            self.tandem.queue[n_s].ql_ave = 0
        self._take_action(action)
        # s_prime 旧状态
        s_prime = self.qls
        delay_vec_dep = []
        observed_delay_vec = []
        # job index in current time slot
        # 计算end-to-end delay
        for index in self.arr_inSlot:
            if index in self.dep_vec[:-1, 0]:
                # ---------if the arrival departs in the same timeslot-----------------
                # 最后一个队列的time departure减去第一个队列的time arrival得到当前job的 end-2-end 延迟
                observed_delay = self.tandem.queue[-1].job_dict[index]['Td'] - self.tandem.queue[0].job_dict[index][
                    'Ta']
                observed_delay_vec.append(observed_delay)
            elif self.t_slot - self.tandem.queue[0].job_dict[index]['Ta'] > self.d_ub:
                # --if the elapsed time spent in this slot is already larger than d_ub-
                observed_delay = self.t_slot - self.tandem.queue[0].job_dict[index]['Ta']
                observed_delay_vec.append(observed_delay)

        for index in self.dep_vec[:-1, 0]:
            delay = self.tandem.queue[-1].job_dict[index]['Td'] - self.tandem.queue[0].job_dict[index]['Ta']
            for n_s in range(len(self.N_s)):
                del (self.tandem.queue[n_s].job_dict[index])
            delay_vec_dep.append(delay)

        reward, num_violation_qos = self._get_reward(np.array(observed_delay_vec), action)

        done = False
        # 检查平均延迟是否超过了预先设定的最大延迟，或者时间步骤是否超过了最大的步骤数
        if np.mean(delay_vec_dep) > self.delay_max or self.cnt > self.MAX_STEPS:
            done = True
        self.cnt += 1 # 时间步骤增加

        # s_prime_bin
        s_prime_bin = []
        # for j in range(len(N_s))
        for j in range(len(self.N_s)):
            s_prime_bin = np.append(s_prime_bin,
                                    np.array(list(np.binary_repr(int(s_prime[j])).zfill(self.B_max))).astype(np.int8)[
                                    :self.B_max])
        # return s_prime_bin, reward, done, delay_vec_dep
        if len(observed_delay_vec)==0:
            delay_avg = 0
        else:
            delay_avg = np.mean(observed_delay_vec)

        # 返回的串联队列的新状态需要额外的剪切步骤，使得队列长度不超过1024
        return np.clip(self.qls,0,1024), reward,done, num_violation_qos/len(self.arr_inSlot), delay_avg

    # 串联队列重置函数，每个episode开始之前需要重置环境
    def reset(self):
        self.t_arr = 0
        self.arr_num_avg = 0
        self.job_index = 1
        self.priority = 10
        self.finish_onestep = False
        self.cnt = 1
        self.qls = np.zeros(len(self.N_s), dtype=int)
        self.dep_vec = []
        self.tandem_job_dict = {}
        self.tandem = Tandem(self.N_s, self.Mu_s)
        self.cost = 0
        self.qls_ave = np.zeros(len(self.N_s), dtype=float)
        self.t_slot = 0
        self.t_arr_vec = []
        qls_bin = np.zeros(self.B_max * len(self.N_s), dtype=np.int8)
        # return qls_bin
        return self.qls

    # 串联队列采取action
    def _take_action(self, action):
        self.dep_vec = []
        flag = True
        # 在当前time slot到达的任务
        self.arr_inSlot = []
        # info_vec 存储两个相邻任务的信息
        # info_vec = np.zeros((2, 3))
        info_vec = np.zeros((2,4))
        ql_init = self.qls
        # take action更改每个队列的 service rate
        for n_s in range(len(self.N_s)):
            self.tandem.queue[n_s].mu_s = action[n_s]

        # 在当前time slot模拟任务到达
        while (True):
            # [job_index, arrival_time, isArrival, priority]
            # info_vec[0] = [self.job_index, self.t_arr, 1]
            # if flag:
            if True:
                self.arr_inSlot.append(self.job_index)
                info_vec[0] = [self.job_index, self.t_arr, 1, self.priority]
                # if self.finish_onestep:
                # 这部分代码主要用于测试队列的规则是按优先权优先服务
                if False:
                    info_vec[1] = [self.job_index, self.t_arr, 0, self.priority]
                    self.finish_onestep = False
            else:
                self.arr_inSlot.append(int(info_vec[1][0]))
                info_vec[0] = [info_vec[1][0],info_vec[1][1],1,info_vec[1][3]]
                flag = True
            # 任务标识+1
            self.job_index += 1
            # t_arr 刚开始为0，然后不断叠加，每次叠加的值来自gamma分布，使得 arrival rate 为 0.95
            # self.t_arr = self.t_arr + self._inter_arr_gen()
            # 生成计算任务的优先权，随机返回一个整数
            interval_time,self.priority = self._inter_arr_gen()
            self.t_arr = self.t_arr + interval_time
            if self.t_arr > self.t_slot + self.T: # 说明一个 time slot 已经完成, 当前任务的到达时间超过了这个time slot
                t_slot_old = self.t_slot
                # time slot 增加
                self.t_slot += self.T
                # info_vec[1] = [self.job_index, self.t_slot, 0]
                info_vec[1] = [self.job_index, self.t_slot, 0, self.priority]
                # 当前time slot的arrivals 收集完毕，模拟串行队列运行，返回所有队列的长度和离开任务
                self.qls, dep_vec = self.tandem._step(info_vec)
                self.dep_vec = np.append(self.dep_vec, dep_vec).reshape(-1, 3) # 矩阵第一维度自适应，第二维度为3
                self.arr_num_avg = (self.arr_num_avg*(self.cnt-1))/self.cnt+(len(self.arr_inSlot)/self.cnt)
                self.finish_onestep = True
                break

            # info_vec[1] = [self.job_index, self.t_arr, 0]
            # if self.priority <= info_vec[0][3]:
            if self.priority <= 20:
                info_vec[1] = [self.job_index, self.t_arr, 0, self.priority]
            else:
                self.arr_inSlot.pop()
                self.arr_inSlot.append(self.job_index)
                info_vec[0] = [self.job_index, self.t_arr, 1, self.priority]
                flag = False

            # 模拟串行队列运行，返回所有队列的长度和离开任务的向量
            self.qls, dep_vec = self.tandem._step(info_vec)
            self.dep_vec = np.append(self.dep_vec, dep_vec[dep_vec[:, 2] == 1]).reshape(-1, 3)

        for i in range(len(self.N_s)):

            coeff = np.append(np.ones(len(self.tandem.arrival_vec_q[i])), -np.ones(len(self.tandem.departure_vec_q[i])))
            arr_dep = np.append(self.tandem.arrival_vec_q[i], self.tandem.departure_vec_q[i])
            ind_sorted = np.argsort(arr_dep)
            # 只要有一个时间点大于当前的time slot就会报错
            assert (not np.sum(arr_dep[ind_sorted] > self.t_slot)), 'arr_dep error'
            arr_dep = np.append(arr_dep[ind_sorted], [self.t_slot])
            coeff = coeff[ind_sorted]
            arr_dep_diff = np.append([arr_dep[0] - (self.t_slot - self.T)], np.diff(arr_dep))
            ql_diff = np.zeros(len(coeff) + 1)
            ql_diff[0] = ql_init[i]

            for j in range(1, len(coeff) + 1):
                ql_diff[j] = max(0, ql_diff[j - 1] + coeff[j - 1])
            self.qls_ave[i] = np.sum(ql_diff * arr_dep_diff) / self.T
            self.tandem.arrival_vec_q[i] = []
            self.tandem.departure_vec_q[i] = []

    # 串联队列系统返回reward的函数
    def _get_reward(self, delay_vec, action):
        # ------- Define reward here--------
        # return r
        mu_sum = 0
        r_t = 0
        # 计数所有违反的计算任务数量
        num_violation_qos = 0
        # 计算服务计算速度的总和
        for i,a in enumerate(action):
            mu_sum = mu_sum + self.N_s[i]*a

        # 计算每个任务的奖励，取决于该任务的端到端延迟是否超过了最大可以忍受的延迟
        for delay in delay_vec:
            # r_i = 0
            if delay <= self.d_ub:
                r_i = self.epsilon_ub*self.tradeoff_lambda
            else:
                r_i = -(1-self.epsilon_ub)*self.tradeoff_lambda
                num_violation_qos += 1
            r_t += r_i
        # return (r_t/len(self.arr_inSlot)) - mu_sum, num_violation_qos
        # return (r_t/self.arr_num_avg) - mu_sum, num_violation_qos
        return (r_t*self.lambda_rate/self.T) - mu_sum, num_violation_qos

    # inter arrivals generation， Gamma函数
    def _inter_arr_gen(self):
        c_a2 = 0.7  # SCV^2
        mean = 1 / self.lambda_rate
        k = 1 / c_a2
        theta = mean / k
        interTa = np.random.gamma(k, theta)  # (shape,scale)
        priority = np.random.randint(1,10) # 优先权生成
        return interTa, priority

    # 用于渲染学习过程，暂时不用实现
    def render(self, mode="human"):
        pass

# 串联队列基础类
class Tandem:
    def __init__(self, N_s, Mu_s):

        self.N_s = N_s
        self.Mu_s = Mu_s
        self.queue = []
        self.ql = np.zeros(len(N_s), dtype=int)
        self.arrival_vec_q = {}
        self.departure_vec_q = {}

        for i, n_s in enumerate(self.N_s):
            self.queue.append(Queue(n_s, self.Mu_s[i]))
            self.arrival_vec_q[i] = []
            self.departure_vec_q[i] = []

    # 串联队列每一步的运行模拟
    def _step(self, info_vec):
        # [job_index, t_arr, isArrival, priority]
        info_vec_new = np.copy(info_vec)
        for i in range(len(self.N_s)): # 逐个处理每个队列
            isArr = (info_vec_new[:, 2] == 1)
            self.arrival_vec_q[i] = np.append(self.arrival_vec_q[i], info_vec_new[isArr, 1]) # 把任务的到达时间点添加到arrival向量中
            # self.arrival_vec_q[i] = np.append(self.arrival_vec_q[i], [info_vec_new[isArr, 1],info_vec_new[isArr,3]])
            self.ql[i], departure_vec = self.queue[i]._progress(info_vec_new) # 模拟每一个队列的运行
            if np.shape(departure_vec)[0] > 0:
                self.departure_vec_q[i] = np.append(self.departure_vec_q[i], departure_vec[:, 1].tolist())
            if np.shape(departure_vec)[0] > 1:
                ind_sorted = np.argsort(departure_vec[:, 1])
                departure_vec = departure_vec[ind_sorted]
            # info_vec_new = np.append(departure_vec, info_vec[-1]).reshape(-1, 3)
            info_vec_new = np.append(departure_vec, info_vec[-1][:-1]).reshape(-1, 3)
        return self.ql, info_vec_new


# 组成串联队列的队列类
class Queue:
    def __init__(self, n_s, mu_s):

        self.n_servers = n_s
        self.n_jobs = 0
        # single queue length
        self.ql_vec = [0]
        # single queue length average
        self.ql_ave = 0
        # 空闲服务器
        self.empty_servers = np.arange(n_s)
        # 已经被分配的服务器
        self.assigned_servers = []
        # t_fin -> time_finish
        self.t_fin = []
        # job_index finish
        self.ind_fin = []
        self.job_dict = {}
        # Tw:waiting time  Ts: service time
        self.job_dict[0] = {'Tw': 0.0, 'Ts': 0.0}
        self.mu_s = mu_s

    # 每运行一布，队列的状态更新
    def _progress(self, info_vec):
        # -----Queue length before taking the action (upon job arrival)---------
        # 当前某个队列的任务离开向量
        departure_vec = []
        assert (np.shape(info_vec)[0] >= 1), 'error'
        for j in range(np.shape(info_vec)[0] - 1):
            job_index = int(info_vec[j][0])
            time = info_vec[j][1]
            isArrival = info_vec[j][2]
            self.ql = max(self.n_jobs - self.n_servers, 0)  # ---before arrival----

            if isArrival:
                if self.n_jobs < self.n_servers:
                    # time enter
                    t_ent = time
                    self.empty_servers = [x for x in range(self.n_servers) if x not in self.assigned_servers]
                    self.assigned_servers = np.append(self.assigned_servers, random.choice(self.empty_servers))

                else:
                    # -------finding the time that each server gets empty---------
                    t_available = [np.max(self.t_fin[self.assigned_servers == i]) for i in range(self.n_servers)]
                    # --------------pick the earliest server available------------
                    picked_server = np.argmin(t_available)
                    # 下一个任务的开始服务时间
                    t_ent = max(time, t_available[picked_server])
                    self.assigned_servers = np.append(self.assigned_servers, picked_server)

                # 生成当前任务的服务时间
                t_s = self._service_gen()
                # t_s = self._exp_service_gen()
                self.t_fin = np.append(self.t_fin, t_ent + t_s)
                self.ind_fin = np.append(self.ind_fin, job_index)
                self.n_jobs += 1
                # time arrival, time departure, service time, time waiting, backlog(堆积在队列中的任务)
                self.job_dict[job_index] = {'Ta': time, 'Td': t_ent + t_s, 'Ts': t_s, 'Tw': t_ent - time,
                                            'Ba': self.ql}

            # 下一个任务的到达时间点
            next_time = info_vec[j + 1][1]
            # 如果有任务可以在下一个任务到达之前就完成，那么任务数减一
            self.n_jobs -= np.sum(np.array(self.t_fin) < next_time)
            served_jobs = np.arange(len(self.t_fin))[np.array(self.t_fin) < next_time]
            for i in served_jobs:
                departure_vec.append([int(self.ind_fin[i]), self.t_fin[i], 1]) # job_index, time_finish, 1

            # 删除已经完成服务的任务，释放资源
            self.t_fin = np.delete(self.t_fin, served_jobs)
            self.ind_fin = np.delete(self.ind_fin, served_jobs)
            self.assigned_servers = np.delete(self.assigned_servers, served_jobs)

        if np.shape(info_vec)[0] == 1:
            next_time = info_vec[0][1]
            self.n_jobs -= np.sum(np.array(self.t_fin) < next_time)
            served_jobs = np.arange(len(self.t_fin))[np.array(self.t_fin) < next_time]
            for i in served_jobs:
                departure_vec.append([int(self.ind_fin[i]), self.t_fin[i], 1])
            self.t_fin = np.delete(self.t_fin, served_jobs)
            self.ind_fin = np.delete(self.ind_fin, served_jobs)
            self.assigned_servers = np.delete(self.assigned_servers, served_jobs)

        # queue length of this stage before the next arrival to the first stage
        QL = max(self.n_jobs - self.n_servers, 0)
        return QL, np.array(departure_vec)

    # service time generation Gamma分布
    def _service_gen(self):
        c_s2 = 0.8  # SCV^2
        mean = 1 / self.mu_s # mu_s 增加，gamma分布的均值就会减小
        k = 1 / c_s2 # shape
        theta = mean / k # scale
        Ts = np.random.gamma(k, theta)
        return Ts

    # 指数分布
    def _exp_service_gen(self):
        exp_lambda = self.mu_s  # 1/mu_s is the mean service time = 1 / lambda
        ts = np.random.exponential(exp_lambda)
        return ts


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# env = gym.make('Pendulum-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 返回初始化权重参数的边界
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=16, fc2_units=16, gru_units=16,gru_fc=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gru_units=gru_units
        self.action_size = action_size
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)
        self.lstm1 = nn.LSTM(1,16,batch_first=True)
        # self.gru1 = nn.GRU(1,self.gru_units,batch_first=True)
        self.fc1 = nn.Linear(self.action_size*self.gru_units,gru_fc)
        self.fc2 = nn.Linear(gru_fc,action_size)
        # self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # print('state shape: ',state.shape)
        # x = F.relu(self.fc1(state))
        # print('xfc1: ',x.shape)
        # x = F.relu(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # print('x shape: ',x.shape)
        # return x
        # print('state shape: ', state.shape)
        # print('state view: ', torch.reshape(state,(-1,2,1)).shape)
        x,_ = self.lstm1(torch.reshape(state,(-1,self.action_size,1)))
        x = F.relu(self.fc1(torch.reshape(x,(-1,self.action_size*self.gru_units))))
        x = torch.tanh(self.fc2(x))
        # print('x1 shape: ',x.shape)
        # x = torch.reshape(x,(-1,))
        # print('x shape: ',x.shape)
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=32, fc2_units=32,gru_units=16,gru_fc=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gru_units=gru_units
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        # self.lstm1 = nn.LSTM(1, 16, batch_first=True)
        # self.gru1 = nn.GRU(1,self.gru_units,batch_first=True)
        # self.fc1 = nn.Linear(2 * self.gru_units, gru_fc)
        # self.fc2 = nn.Linear(action_size,gru_fc)
        # self.fc3 = nn.Linear(gru_fc+gru_fc,1)
        # self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # print('state: ',state.shape)
        xs = F.relu(self.fcs1(state))
        # print('xs: ',xs.shape)
        # print('action: ',action.shape)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        # xs, _ = self.lstm1(torch.reshape(state,(-1,2,1)))
        # xs = F.relu(self.fc1(torch.reshape(xs, (-1, 2 * self.gru_units))))
        # xa = F.relu(self.fc2(action))
        # x = torch.cat((xs,xa),dim=1)
        # x = self.fc3(x)
        # return x

# 衰减的高斯过程，主要用于采样衰减的随机噪声
class AnnealedGaussianProcess():
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma
# OU噪声生成
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, seed, mu=0., sigma=1., dt=1e-2, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset(self):
        self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size)
# 无衰减的OU噪声
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

# 经验缓存
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    # 添加经验元组
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # 采样一批experience
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# DDPG Agent
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed,add_noise=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.add_noise = add_noise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        # self.noise = OUNoise(action_size, random_seed)
        self.noise = OrnsteinUhlenbeckProcess(theta=0.15,seed=random_seed,sigma=0.5,sigma_min=0.05,size=action_size,n_steps_annealing=53200)
        # self.noise = OrnsteinUhlenbeckProcess(theta=0.15, seed=random_seed, sigma=0.15,size=action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            action = action[0]
        self.actor_local.train()
        # scale action value from [-1,1] to action domain
        scale_action = np.multiply((action+1)/2,action_domain)+lower_bound
        if self.add_noise:
            scale_action += self.noise.sample()
        # return np.clip(action, -1, 1)
        return np.clip(scale_action,lower_bound,upper_bound)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# Baseline
class SimpleAgent():
    def __init__(self,random_seed):
        self.random_seed = random.seed(random_seed)
        self.max_ql = 1024
        self.add_noise = False

    def act(self,state):
        a = state[0] / 1024 * 2
        if a == 0:
            a = lower_bound[0]*2

        b = state[1] / 1024 * 2
        if b == 0:
            b = lower_bound[1]*2

        c = state[2] / 1024 * 2
        if c == 0:
            c = lower_bound[2]*2
        return np.array([a,b,c])

    def reset(self):
        pass


label_size = 11
ticker_size = 10
def plot_three_metrics(scores,episodes_sum_rates,episodes_violation_probas,episodes_delays,i_episode,save_fig=False):
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4.4))
    fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(19, 4.4),linewidth=0)
    # ax1.plot(np.arange(1, len(episodes_violation_probas) + 1), episodes_violation_probas,'#2c7eec')
    # ax1.hlines(y=env.epsilon_ub, xmin=1, xmax=len(episodes_violation_probas), linewidth=2, color='r', ls='--')
    # ax1.annotate(r'$\varepsilon _{ub}=0.1$',
    #              xy=(100, 0.1), xycoords='data',
    #              xytext=(99.4, 0.2), textcoords='data',
    #              arrowprops=dict(width=5, color='k', ),
    #              size=15
    #              )
    # ax1.set(xlabel='Episodes', ylabel=r'$P\left(d>d_{ub}\right)$')
    # ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    ax2.plot(np.arange(1, len(episodes_sum_rates) + 1), episodes_sum_rates,'#2c7eec')
    # ax2.set(xlabel='Episodes', ylabel='Sum Computation Speed')
    ax2.set_xlabel('Episodes', fontsize=label_size)
    ax2.set_ylabel('Sum Computation Speed', fontsize=label_size)
    ax2.tick_params(axis="x", labelsize=ticker_size)
    ax2.tick_params(axis="y", labelsize=ticker_size)
    ax2.set_yticks([2,4, 6, 8, 10,12,14,16])
    # ax2.set_yticks([2, 8, 14, 20, 26])


    ax3.plot(np.arange(1, len(scores) + 1), scores,'#2c7eec')
    # ax3.set(xlabel='Episodes', ylabel='Average Reward')
    ax3.set_xlabel('Episodes', fontsize=label_size)
    ax3.set_ylabel('Average Reward', fontsize=label_size)
    ax3.tick_params(axis="x", labelsize=ticker_size)
    ax3.tick_params(axis="y", labelsize=ticker_size)
    ax3.set_yticks([-20,-18,-16,-14,-12, -10, -8, -6])

    ax4.plot(np.arange(1,len(episodes_delays)+1),episodes_delays,'#2c7eec')
    # ax4.set(xlabel='Episodes', ylabel='Average Delay')
    ax4.set_xlabel('Episodes', fontsize=label_size)
    ax4.set_ylabel('Average Delay', fontsize=label_size)
    ax4.tick_params(axis="x", labelsize=ticker_size)
    ax4.tick_params(axis="y", labelsize=ticker_size)
    ax4.set_yticks([2,4,6,8])

    # if save_fig:
    #     noise_name = 'without noise'
    #     if agent.add_noise:
    #         noise_name = 'with noise'
    #     fig_name = noise_name + '_lambda' + str(env.tradeoff_lambda) + '_step' + str(env.T) + '_episode' + str(
    #         i_episode) + '.pdf'
    #     fig.savefig(fig_name, format='pdf',bbox_inches='tight')
    extent_ax2 = ax2.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(str(i_episode)+'sumComputationSpeed_fig.pdf', bbox_inches=extent_ax2)
    extent_ax3 = ax3.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(str(i_episode)+'averageReward_fig.pdf', bbox_inches=extent_ax3)
    extent_ax4 = ax4.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(str(i_episode)+'averageDelay_fig.pdf', bbox_inches=extent_ax4)
    plt.show()

# 画出实验图，带error band
def plot_error_band(all_scores,all_episodes_sum_rates, all_episodes_violation_probas, all_episodes_delays):
    a = np.array(all_scores) # rewards
    b = np.array(all_episodes_sum_rates) # sum-rates
    # c = np.array(all_episodes_violation_probas) # violation probability
    d = np.array(all_episodes_delays) # delay

    a_mean = np.mean(a,axis=0)
    a_std = np.std(a,axis=0)

    b_mean = np.mean(b,axis=0)
    b_std = np.std(b,axis=0)

    # c_mean = np.mean(c,axis=0)
    # c_std = np.std(c,axis=0)

    d_mean = np.mean(d,axis=0)
    d_std = np.std(d,axis=0)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4.4))
    fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(19, 4.4),linewidth=0)
    # ax1.plot(np.arange(1, len(c_mean) + 1), c_mean,'#2c7eec')
    # ax1.fill_between(np.arange(1, len(c_mean) + 1),c_mean-c_std,c_mean+c_std,color='#2c7eec',alpha=0.5)
    # ax1.hlines(y=env.epsilon_ub, xmin=1, xmax=len(c_mean), linewidth=2, color='r', ls='--')
    # ax1.annotate(r'$\varepsilon _{ub}=0.1$',
    #              xy=(100, 0.1), xycoords='data',
    #              xytext=(99.4, 0.2), textcoords='data',
    #              arrowprops=dict(width=5, color='k', ),
    #              size=15
    #              )
    # ax1.set(xlabel='Episodes', ylabel=r'$P\left(d>d_{ub}\right)$')
    # ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    ax2.plot(np.arange(1, len(b_mean) + 1), b_mean,'#4285F4')
    ax2.fill_between(np.arange(1, len(b_mean) + 1), b_mean - b_std, b_mean + b_std, color='#4285F4', alpha=0.3)
    # ax2.set(xlabel='Episodes', ylabel='Sum Computation Speed')
    ax2.set_xlabel('Episodes', fontsize=label_size)
    ax2.set_ylabel('Sum Computation Speed', fontsize=label_size)
    ax2.tick_params(axis="x", labelsize=ticker_size)
    ax2.tick_params(axis="y", labelsize=ticker_size)
    ax2.set_yticks([2, 4, 6, 8, 10,12,14,16])
    # ax2.set_yticks([2, 8, 14, 20, 26])

    ax3.plot(np.arange(1, len(a_mean) + 1), a_mean,'#4285F4')
    ax3.fill_between(np.arange(1, len(a_mean) + 1), a_mean - a_std, a_mean + a_std, color='#4285F4', alpha=0.3)
    # ax3.set(xlabel='Episodes', ylabel='Average Reward')
    ax3.set_xlabel('Episodes', fontsize=label_size)
    ax3.set_ylabel('Average Reward', fontsize=label_size)
    ax3.tick_params(axis="x", labelsize=ticker_size)
    ax3.tick_params(axis="y", labelsize=ticker_size)
    ax3.set_yticks([-20,-18,-16,-14,-12, -10, -8, -6])

    ax4.plot(np.arange(1,len(d_mean)+1),d_mean, '#4285F4')
    ax4.fill_between(np.arange(1,len(d_mean)+1),d_mean-d_std, d_mean+d_std, color='#4285F4',alpha=0.3)
    # ax4.set(xlabel='Episodes',ylabel='Average Delay')
    ax4.set_xlabel('Episodes', fontsize=label_size)
    ax4.set_ylabel('Average Delay', fontsize=label_size)
    ax4.tick_params(axis="x", labelsize=ticker_size)
    ax4.tick_params(axis="y", labelsize=ticker_size)
    ax4.set_yticks([2,4,6,8])

    # noise_name = 'without noise'
    # if agent.add_noise:
    #     noise_name = 'with noise'
    # fig_name = 'error_band_'+noise_name + '_lambda' + str(env.tradeoff_lambda) + '_step' + str(env.T) + '_episodes' + str(
    #     n_episodes) + '.pdf'
    # fig.savefig(fig_name, format='pdf',bbox_inches='tight')
    extent_ax2 = ax2.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('error band'+str(n_episodes)+'sumComputationSpeed_fig.pdf', bbox_inches=extent_ax2)
    extent_ax3 = ax3.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('error band'+str(n_episodes)+'averageReward_fig.pdf', bbox_inches=extent_ax3)
    extent_ax4 = ax4.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('error band'+str(n_episodes)+'averageDelay_fig.pdf', bbox_inches=extent_ax4)
    plt.show()

BUFFER_SIZE = int(1e5)  # replay buffer size # 1e5
BATCH_SIZE = 128      # minibatch size 128
GAMMA = 0.99            # discount factor
TAU = 1e-2             # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 1e-3
LR_CRITIC = 1e-3       # learning rate of the critic 0.005
WEIGHT_DECAY = 0        # L2 weight decay
n_episodes = 1200


def ddpg(n_episodes=n_episodes, max_t=300, print_every=100,agent_type='ddpg'):
    # scores_deque = deque(maxlen=print_every) # 保存最新的100个 episode reward值
    scores = []
    episodes_sum_rates = []
    episodes_violation_probas = []
    episodes_delays = []
    all_states = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset()
        step_rewards = []
        step_sum_rates = []
        step_violation_probas = []
        step_delays = []
        # for t in range(max_t):
        #     action = agent.act(state)
        #     next_state, reward, done, _ = env.step(action)
        #     agent.step(state, action, reward, next_state, done)
        #     state = next_state
        #     score += reward
        #     if done:
        #         break
        while True:
            action = agent.act(state)
            sum_rate = np.sum([a*env.N_s[i] for i,a in enumerate(action)])
            next_state,reward,done,qos_violation_proba, step_delay = env.step(action)
            all_states.append(next_state)
            if agent_type == 'ddpg':
                agent.step(state,action,reward,next_state,done)
            state = next_state
            step_rewards.append(reward)
            step_sum_rates.append(sum_rate)
            step_violation_probas.append(qos_violation_proba)
            step_delays.append(step_delay)
            if done:
                break
        step_rewards_avg = np.mean(step_rewards)
        step_sum_rates_avg = np.mean(step_sum_rates)
        step_violation_probas_avg = np.mean(step_violation_probas)
        step_delays_avg = np.mean(step_delays)
        # scores_deque.append(step_rewards_avg)
        scores.append(step_rewards_avg)
        episodes_sum_rates.append(step_sum_rates_avg)
        episodes_violation_probas.append(step_violation_probas_avg)
        episodes_delays.append(step_delays_avg)
        print('Episode {}/{}  Violation Proba: {:.2f}  Sum-Rate: {:.2f}  Average Reward: {:.2f}  Average Delay: {:.2f}'.format(i_episode, n_episodes, step_violation_probas_avg, step_sum_rates_avg, step_rewards_avg,step_delays_avg))
        if i_episode % 50 == 0:
            torch.save(agent.actor_local.state_dict(), str(i_episode)+'episodes_checkpoint_actor_'+str(env.d_ub)+'_'+str(env.tradeoff_lambda)+'lstm'+'.pth')
            torch.save(agent.critic_local.state_dict(), str(i_episode)+'episodes_checkpoint_critic_'+str(env.d_ub)+'_'+str(env.tradeoff_lambda)+'lstm'+'.pth')
                # print('\rEpisode {}\tAverage Reward: {:.2f}'.format(i_episode, np.mean(scores_deque)))

            plot_three_metrics(scores, episodes_sum_rates, episodes_violation_probas, episodes_delays,i_episode, save_fig=True)

    return scores,episodes_sum_rates,episodes_violation_probas, episodes_delays, all_states

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

total_rounds = 3
agent_type = 'ddpg'
# agent_type = 'simple'
random_seeds = np.random.randint(1,100,total_rounds)
# random_seeds = [7,11,8,26,97]
all_scores = []
all_episodes_sum_rates = []
all_episodes_violation_probas = []
all_episodes_delays = []

for i,random_seed in enumerate(random_seeds[:total_rounds]):
    print('\n----------------------------------------------------------')
    print('Agent: '+ agent_type)
    print('Round: {}/{}'.format(i+1,total_rounds))
    print('Random seed: {}'.format(random_seed))
    # N_s, Mu_s, lambda_rate, d_ub, step_len, random_seed, tradeoff_lambda
    # env = TandemEnv([3, 5], [0.35, 0.21], 0.95, 5, 15, random_seed,12)
    env = TandemEnv([3], [0.35], 0.95, 5, 15, random_seed, 12)
    # print('state dim:'+str(env.action_space.shape[0]))
    if agent_type == 'ddpg':
        agent = Agent(state_size=len(env.N_s), action_size=len(env.N_s), random_seed=random_seed, add_noise=True)
    else:
        agent = SimpleAgent(random_seed)
    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    action_domain = upper_bound - lower_bound
    if i == 0:
        print("Max Value of Action ->  {}".format(upper_bound))
        print("Min Value of Action ->  {}".format(lower_bound))
        print("Range of Action value -> {}".format(action_domain))
        print("Tradeoff Lambda -> {}, Step Length -> {}, With Noise -> {}".format(env.tradeoff_lambda, env.T,agent.add_noise))
    print('----------------------------------------------------------\n')
    scores, episodes_sum_rates, episodes_violation_probas, episodes_delays, all_states = ddpg(agent_type=agent_type)
    all_scores.append(scores)
    all_episodes_sum_rates.append(episodes_sum_rates)
    all_episodes_violation_probas.append(episodes_violation_probas)
    all_episodes_delays.append(episodes_delays)
    all_states = np.array(all_states)
    states_max = np.max(all_states,axis=0)
    states_min = np.min(all_states,axis=0)
    states_mean = np.mean(all_states,axis=0)
    print('--------------------------------------------------')
    print(states_max)
    print(states_min)
    print(states_mean)
    print('--------------------------------------------------')

plot_error_band(all_scores,all_episodes_sum_rates,all_episodes_violation_probas, all_episodes_delays)

# start_time = time.time()
# scores,episodes_sum_rates,episodes_violation_probas = ddpg()
# print("\n--- %s seconds ---" % (time.time() - start_time))
# plot_three_metrics(scores,episodes_sum_rates,episodes_violation_probas,save_fig=True)

