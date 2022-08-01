import numpy as np
import matplotlib.pyplot as plt

from ddpg_Qmatrix_lib import DDPG

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 5
MEMORY_CAPACITY = 100
BOUND = 0.2

###############################  DDPG  ####################################
if __name__=='__main__':

    a_dim = 3  
    s_dim = 6

    reward_curve = np.zeros(MAX_EPISODES)
    best_reward = np.zeros(1)
    best_Q_matirx = np.zeros(s_dim)
    _matrix = np.zeros(s_dim)

    ddpg = DDPG(a_dim, s_dim)

    var = 3  # control exploration
    for i in range(MAX_EPISODES):
#        s = env.reset()
        location = np.array([(-2707029.10975552, 4688711.95566451, 3360431.43412232),
                            (-2707028.10975552, 4688711.95566451, 3360431.43412232),
                            (-2707027.10975552, 4688711.95566451, 3360431.43412232),
                                (-2707026.10975552, 4688711.95566451, 3360431.43412232),
                            (-2707025.10975552, 4688711.95566451, 3360431.43412232)])
        s = np.array([-524645.254430573, 908711.849831743, 651279.822781636, 0.0, 0.0, -0.979484197226504])
        #s = np.array([-524645.423231112, 908711.681031204, 651279.653981097, 0.0, 0.0, -0.979484197226504])
        #s = np.array([-524645.396906058, 908711.707356257, 651279.680306151, 0.0, 0.0, -0.979484197226504])

        initial_error = (((location[0][0] - s[0]) ** 2) + ((location[0][1] - s[1]) ** 2) + ((location[0][2] - s[2]) ** 2)) ** 0.5
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
#            if RENDER:
#                env.render()
            location_error = (((location[j][0] - s[0]) ** 2) + ((location[j][1] - s[1]) ** 2) + ((location[j][2] - s[2]) ** 2)) ** 0.5
            for k in range(3):
                s[k] = s[k] / 1000000

            a = ddpg.choose_action(s).numpy()
            #print('Episode:', i, "a = ",a)
            a[0] = np.clip(np.random.normal(a[0], var), a[0] - BOUND, a[0] + BOUND)    # np.random.normal(mean,std) 表示爲一個正態分佈 np.clip表示Limit the value between -2 and 2
#            s_, r, done, info = env.step(a)
            rand_num = np.random.randint(a_dim, size = 1)
            #print('Episode:', i, "rand_num = ", rand_num)
            rand_a = a[rand_num]
            #Sprint('Episode:', i, "rand_a = ", rand_a)
            #s_ = s * a[0] - 10 * a[1]
            s_ = s * (a[0] ** 2) - 10 * a[1] - a[2]
            #print('Episode:', i, "s_ = ", s_)
            r = -location_error

        
            #將資料存入self.pointer = i的位置
            ddpg.store_transition(s, a, r, s_)
            if ddpg.pointer > MEMORY_CAPACITY:     # wait for the memory pool being full at first
                var *= 0.999    # decay the action randomness
                ddpg.learn()

            ep_reward += r
            if j == MAX_EP_STEPS-1:
                #print('Episode:', i, ' Reward: %i' % int(ep_reward))
                #print("Episode:", (i + 1), ", Reward = ", r, ", position = ", s[0 : 3] * 1000000)
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
