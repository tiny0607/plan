import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 5
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9    # reward discount
TAU = 0.01    # soft replacement
MEMORY_CAPACITY = 100
BATCH_SIZE = 32
BOUND = 0.2

###############################  DDPG  ####################################

class ANet(nn.Module):   # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1) 
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1) 
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x * BOUND  # for the game "Pendulum-v0", action range is [-2, 2]
        return actions_value

class CNet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1) 
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1) 
    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.a_dim, self.s_dim,  = a_dim, s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype = np.float32) # s,s_,a,r
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr = LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr = LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        #FloatTensor將初始的s值([3])匯入轉為torch形式(float值)，並用unsqueeze在0的位置增加一維度變為[1, 3]
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        #返回新的tensor，detach()使其不會有梯度grad
        return self.Actor_eval(s)[0].detach() # ae（s）

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1 - TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU * self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1 - TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU * self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        indices = np.random.choice(MEMORY_CAPACITY, size = BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一個正確的行爲的話，那麼它的Q應該更貼近0
        loss_a = -torch.mean(q) 
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 這個網絡不及時更新參數, 用於預測 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)  # 這個網絡不及時更新參數, 用於給出 Actor 更新參數時的 Gradient ascent 強度
        q_target = br + (GAMMA * q_)  # q_target = 負的
        #print(q_target)
        q_v = self.Critic_eval(bs, ba)
        #print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但這個ae(s)是記憶中的ba，讓ce得出的Q靠近Q_target,讓評價更準確
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

def ddpg_Qmatrix(ini_location, ini_s):
    a_dim = 3
    s_dim = 6

    reward_curve = np.zeros(MAX_EPISODES)
    best_reward = np.zeros(1)
    best_Q_matirx = np.zeros(s_dim)
    Q_matrix = np.zeros(s_dim)

    ddpg = DDPG(a_dim, s_dim)


    var = 3  # control exploration
    bound = 0.1
    for i in range(MAX_EPISODES):

        location = ini_location
        s = ini_s

        initial_error = (((location[0][0] - s[0]) ** 2) + ((location[0][1] - s[1]) ** 2) + ((location[0][2] - s[2]) ** 2)) ** 0.5
        ep_reward = 0
        for j in range(MAX_EP_STEPS):

            location_error = (((location[j][0] - s[0]) ** 2) + ((location[j][1] - s[1]) ** 2) + ((location[j][2] - s[2]) ** 2)) ** 0.5
            for k in range(3):
                s[k] = s[k] / 1000000

            a = ddpg.choose_action(s).numpy()
            a[0] = np.clip(np.random.normal(a[0], var), a[0] - bound, a[0] + bound)    # np.random.normal(mean,std) 表示爲一個正態分佈 np.clip表示Limit the value between -2 and 2

            rand_num = np.random.randint(a_dim, size = 1)

            rand_a = a[rand_num]

            s_ = s * (a[0] ** 2) - 10 * a[1] - a[2]
    
            r = -location_error

        
        #將資料存入self.pointer = i的位置
            ddpg.store_transition(s, a, r, s_)
            if ddpg.pointer > MEMORY_CAPACITY:     # wait for the memory pool being full at first
                var *= 0.999    # decay the action randomness
                ddpg.learn()

            ep_reward += r
            if j == MAX_EP_STEPS-1:
                reward_curve[i] = r
                if i == 0:
                    best_reward = r
                    best_Q_matirx = s[0:6]
                if r >= best_reward:
                    best_reward = r
                    best_Q_matirx = s[0:6]
                if ep_reward > -300:RENDER = True
                break
            s = s_
            for k in range(3):
                s_[k] = s_[k] * 1000000

    print("initial_error = ", initial_error)
    print("best_reward = ", best_reward, ", best_Q_matrix = ", best_Q_matirx)
    mean = np.mean(best_Q_matirx)
    Q_matrix = best_Q_matirx / mean
    print("Q_matrix = ", Q_matrix)
    plt.plot(np.linspace(1, MAX_EPISODES, MAX_EPISODES), reward_curve)
    plt.title('Reward curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    return Q_matrix