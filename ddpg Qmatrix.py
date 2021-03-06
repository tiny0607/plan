import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################

MAX_EPISODES = 40
MAX_EP_STEPS = 5
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100
BATCH_SIZE = 32
RENDER = False
#ENV_NAME = 'Pendulum-v1'

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
        actions_value = x * 2  # for the game "Pendulum-v0", action range is [-2, 2]
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
        #FloatTensor????????????s???([3])????????????torch??????(float???)?????????unsqueeze???0??????????????????????????????[1, 3]
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        #????????????tensor???detach()?????????????????????grad
        return self.Actor_eval(s)[0].detach() # ae???s???

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
        #self.sess.run(self.soft_replace)  # ???ae???ce??????at???ct

        indices = np.random.choice(MEMORY_CAPACITY, size = BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce???s,ae???s????????????ae   ae???s???=a   ae???s_???=a_
        # ?????? a?????????????????????????????????????????????Q???????????????0
        loss_a = -torch.mean(q) 
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # ?????????????????????????????????, ???????????? Critic ??? Q_target ?????? action
        q_ = self.Critic_target(bs_, a_)  # ?????????????????????????????????, ???????????? Actor ?????????????????? Gradient ascent ??????
        q_target = br + (GAMMA * q_)  # q_target = ??????
        #print(q_target)
        q_v = self.Critic_eval(bs, ba)
        #print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct???bs_,at(bs_)???-ce(s,ba) ??????ce ,?????????ae(s)???????????????ba??????ce?????????Q??????Q_target,??????????????????
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()


###############################  training  ####################################
'''env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
'''
reward_curve = np.zeros(MAX_EPISODES)

a_dim = 1
s_dim = 6

ddpg = DDPG(a_dim, s_dim)


var = 3  # control exploration
for i in range(MAX_EPISODES):
#    s = env.reset()
    location = np.array([-2707029.10975552, 4688711.95566451, 3360431.43412232])
    s = np.array([-524645.254430573, 908711.849831743, 651279.822781636, 0.0, 0.0, -0.979484197226504])
    initial_error = (((location[0] - s[0]) ** 2) + ((location[1] - s[1]) ** 2) + ((location[2] - s[2]) ** 2)) ** 0.5
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
#        if RENDER:
#            env.render()
        location_error = (((location[0] - s[0]) ** 2) + ((location[1] - s[1]) ** 2) + ((location[2] - s[2]) ** 2)) ** 0.5
        #??????action???
        # Add exploration noise
        a = ddpg.choose_action(s)
        #print('Episode:', i, "a = ",a)
        a = np.clip(np.random.normal(a, var), -2, 2)    # np.random.normal(mean,std) ??????????????????????????? np.clip??????Limit the value between -2 and 2
#        s_, r, done, info = env.step(a)
        rand_num = np.random.randint(a_dim, size = 1)
        #print('Episode:', i, "rand_num = ", rand_num)
        rand_a = a[rand_num]
        #print('Episode:', i, "rand_a = ", rand_a)
        s_ = s * (np.exp(rand_a))
        #print('Episode:', i, "s_ = ", s_)
        r = -location_error
        
        #???????????????self.pointer = i?????????
        ddpg.store_transition(s, a, r, s_)
        if ddpg.pointer > MEMORY_CAPACITY:     # wait for the memory pool being full at first
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        #location = location * (np.exp(rand_a))
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            #print('Episode:', i, ' Reward: %i' % int(ep_reward))
            print("Episode:", (i + 1), "Reward = ", ep_reward)
            reward_curve[i] = r
            if ep_reward > -300:RENDER = True
            break

print("initial_error = ", initial_error)
plt.plot(np.linspace(1, MAX_EPISODES, MAX_EPISODES), reward_curve)
plt.title('Reward curve')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()