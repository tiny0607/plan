# # Reference
# https://www.youtube.com/watch?v=AvKSPZ7oyVg

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pynvml import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

file_path = os.path.abspath('.//code//Research-project//gnss_ins_sim_master//demo_saved_data//')
motion_level = "RP_l1_low_accuracy_long"  # RP_l1 ~ RP_l6
gps_imu_data_file = [file_path + "//" + motion_level + os.path.abspath('//accel-0.csv'),\
                        file_path + "//" + motion_level + os.path.abspath('//gyro-0.csv'),\
                        file_path + "//" + motion_level + os.path.abspath('//mag-0.csv'),\
                        file_path + "//" + motion_level + os.path.abspath('//gps-0.csv')]
IMU_GPS_SMPL_RATIO = 100    # ratio of sample count of IMU over ones of GPS
SMPL_SET = 80          # how many batches separated from data set, must larger than the sum of (TRAIN_SZ+VAL_SZ)
TRAIN_SZ = 64
VAL_SZ = 16      # train:val = 80:20
######################## gradient accumulation ##########################
ITERATION_TIME = 2
GPU_MINI_BATCH_SZ = int(TRAIN_SZ/ITERATION_TIME)
#########################################################################
GPS_SEQ_LEN = 200
SAMPLING_INTERVAL = 10  # sampling interval is 2 samples

# # # Generate data
#######################################################
class Sys_Init():
    # def __init__(self):
    
    def Initialize_Sys(self):
        data_type = "gps_imu_data"
        in_sz = 9    # x_acc x_gyro x_mag y_acc y_gyro y_mag z_acc z_gyro z_mag 
        out_sz = 3   # delta (p_x p_y p_z v_x v_y v_z)
        hidden_sz = 20
    
        return data_type, in_sz, out_sz, hidden_sz
    
    def Use_GPU(self):
        return torch.cuda.is_available()

    def Get_Device(self, use_gpu):
        return torch.device("cuda:0" if use_gpu else "cpu")
    
    def Get_Memory_Info(self):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'total   :{info.total}')
        print(f'free    :{info.free}')
        print(f'used    :{info.used}')        

class Data_Generator():
    def __init__(self, data_type, data_src):
        self.data_type = data_type
        self.data_src = data_src    # list, store 4 files

    """ Generate data in this form (seq_len * batch_sz * feature_sz)"""
    def gen_data(self):
        ### Get imu and gps data from .csv files
        try:
            acc_data = np.genfromtxt(self.data_src[0], delimiter=',', skip_header=1,\
                                        max_rows=self.__max_row(0), dtype=np.float32)
            gyro_data = np.genfromtxt(self.data_src[1], delimiter=',', skip_header=1,\
                                        max_rows=self.__max_row(0), dtype=np.float32)
            mag_data = np.genfromtxt(self.data_src[2], delimiter=',', skip_header=1,\
                                        max_rows=self.__max_row(0), dtype=np.float32)
            gps_data = np.genfromtxt(self.data_src[3], delimiter=',', skip_header=2,\
                usecols=(9,10,11), max_rows=self.__max_row(1), dtype=np.float32)
        except:
            raise ValueError('motion definition file/string must have nine columns \
                            and at least four rows (two header rows + at least two data rows).')
        
        ### Slice data into "SMPL_SET" batches
        return self.__split_sequence(acc_data, gyro_data, mag_data, gps_data)

    def __split_sequence(self, temp_acc_data, temp_gyro_data, temp_mag_data, temp_gps_data):
        acc_data, gyro_data, mag_data, gps_data = list(), list(), list(), list()
        for i in range(SMPL_SET):
            j = i*SAMPLING_INTERVAL
            acc_data.append(temp_acc_data[(j*IMU_GPS_SMPL_RATIO):((j+GPS_SEQ_LEN)*IMU_GPS_SMPL_RATIO), :])    # forward 100 samples per loop
            gyro_data.append(temp_gyro_data[(j*IMU_GPS_SMPL_RATIO):((j+GPS_SEQ_LEN)*IMU_GPS_SMPL_RATIO), :])    # forward 100 samples per loop
            mag_data.append(temp_mag_data[(j*IMU_GPS_SMPL_RATIO):((j+GPS_SEQ_LEN)*IMU_GPS_SMPL_RATIO), :])    # forward 100 samples per loop
            gps_data.append(temp_gps_data[j:(j+GPS_SEQ_LEN),:])
        acc_data = np.array(acc_data)       # concatenate data
        gyro_data = np.array(gyro_data)
        mag_data = np.array(mag_data)
        gps_data = np.array(gps_data)

        acc_data = np.transpose(acc_data, (1,0,2))  # (seq_len, batch, feature) -> (batch, seq_len, feature)
        gyro_data = np.transpose(gyro_data, (1,0,2))
        mag_data = np.transpose(mag_data, (1,0,2))
        gps_data = np.transpose(gps_data, (1,0,2))
        imu_data = np.concatenate([acc_data, gyro_data, mag_data], axis=2) # (batch, seq_len, features)
        
        # Exchange the column forms from xyzxyzxyz to xxxyyyzzz
        # Cluster x-, y- and z-axis data from acc, gyro and mag, respectively
        imu_data[:, :, [0,1,2,3,4,5,6,7,8]] = imu_data[:, :, [0,3,6,1,4,7,2,5,8]]
        train_input = torch.from_numpy(imu_data[:, :TRAIN_SZ, :])
        train_label = torch.from_numpy(gps_data[:, :TRAIN_SZ, :])
        val_input = torch.from_numpy(imu_data[:, TRAIN_SZ:(TRAIN_SZ+VAL_SZ), :])
        val_label = torch.from_numpy(gps_data[:, TRAIN_SZ:(TRAIN_SZ+VAL_SZ), :])

        return train_input, train_label, val_input, val_label

    def __max_row(self, imu_gps=0):
        if "RP_l1_low_accuracy" == motion_level:
            if 0 == imu_gps:        # imu
                return 410000
            else:                   # gps
                return 4100
        elif "RP_l1_low_accuracy_long" == motion_level:
            if 0 == imu_gps:        # imu
                return 1310000
            else:                   # gps
                return 13100
        elif "RP_l1_low_accuracy_long2" == motion_level:
            if 0 == imu_gps:        # imu
                return 2700000
            else:                   # gps
                return 27000
        else:
            return False

""" Generate LSTM module """
class LSTMPredictor(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, device):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        # Output                                                        O
        # Linear
        # LSTM2  .......... .......... .......... .......... .......... .
        # LSTM1  .......... .......... .......... .......... .......... .
        # Input  I      -> Prop. direction
        self.lstm1 = nn.LSTMCell(self.n_in, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm4 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm5 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm6 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, self.n_out)

    # future=0 means it doesn't predict any value, just training. If future=10, it means it predict 10 future points
    def forward(self, x, future=0):
        outputs = []
        batch_sz = x.size(1)
        dim = 0

        # Cell state stores long-term memory; hidden state stores working memory
        h_t = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)    # hidden state. Size is (n_sample * n_hidden)
        c_t = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)    # cell state
        h_t2 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        c_t2 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        h_t3 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        c_t3 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        h_t4 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        c_t4 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        h_t5 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        c_t5 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        h_t6 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)
        c_t6 = torch.zeros(batch_sz, self.n_hidden, dtype=torch.float32, device=self.device)

        # Every cells run once in every iteration to nudge h_t and c_t
        """ Training """
        i = 0
        for time_step in x.split(1, dim=dim):
            time_step = time_step.squeeze()
            h_t, c_t = self.lstm1(time_step, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5, c_t5))
            h_t6, c_t6 = self.lstm6(h_t5, (h_t6, c_t6))
            i += 1
            if IMU_GPS_SMPL_RATIO == i:        # 100 imu data generate 1 gps data
                i = 0
                output = self.linear(h_t6)      # linearly transform data from dim=batch_sz*hidden_sz to dim=batch_sz*n_out
                outputs.append(output)          # append current result to previous ones
        
        """ Prediction """
        for i in range(future):     # future!=0
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        
        outputs = torch.stack(outputs, 0)

        return outputs

class Plot_Loss():
    def __init__(self, loss=torch.tensor([]), init_loss=0):
        self.train_losses = loss
        self.val_losses = loss
        self.init_training_loss = init_loss
    
    def restore_loss_from_optimizer(self, opt_loss, training=True):
        if True == training:       # training loss
            self.train_losses = torch.cat((self.train_losses, torch.tensor([opt_loss])), dim=0)
        else:                      # validation loss
            self.val_losses = torch.cat((self.val_losses, torch.tensor([opt_loss])), dim=0)

    def plot_losses(self):
        n1 = self.train_losses.size(0)
        # Interpolate val_loss to the size of train_losses
        self.val_losses = self.val_losses.unsqueeze(0).unsqueeze(0)
        self.val_losses = nn.functional.interpolate(self.val_losses, size=n1, mode="linear")\
                            .squeeze()[2:]  # Discard the non-linear very beginning
        n2 = self.val_losses.size(0)
        plt.plot(np.arange(n1), self.train_losses[:n1].detach().numpy(), 'r', linewidth=2.0, label="training loss")
        plt.plot(np.arange(n2), self.val_losses[:n2].detach().numpy(), 'g', linewidth=2.0, label="validation loss")
        plt.xlabel("time step")
        plt.ylabel("loss")
        plt.title("Learning Curve", fontsize=20)
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()
    
if __name__ == "__main__":

    m_sys = Sys_Init()
    is_gpu = m_sys.Use_GPU()
    device = m_sys.Get_Device(is_gpu)
    gen_data_type, input_sz, output_sz, hidden_sz = m_sys.Initialize_Sys()
    # Generate simulation data
    m_data_gen = Data_Generator(data_type=gen_data_type, data_src=gps_imu_data_file)
    train_input, train_label, val_input, val_label = m_data_gen.gen_data()
    if True == is_gpu:
        train_input, train_label, val_input, val_label = \
            train_input.to(device), train_label.to(device), val_input.to(device), val_label.to(device)
######################## gradient accumulation ##########################
    mini_train_input = torch.split(train_input, GPU_MINI_BATCH_SZ, dim=1)
    mini_train_label = torch.split(train_label, GPU_MINI_BATCH_SZ, dim=1)
#########################################################################

    model = LSTMPredictor(input_sz, output_sz, hidden_sz, device)
    if True == is_gpu:
        model = model.to(device)
    criterion = nn.MSELoss()
    # Use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=5) # weight decay, learning rate
    plt_loss = Plot_Loss()
    
    n_epoch = 10    # More training times gets better results
    for i in range(n_epoch):
        print("Step", i)

        """ Training """
        model.train()    # Enter training mode
######################## gradient accumulation ##########################
        def closure():
            optimizer.zero_grad()           # Empty the gradient
            train_losses = torch.zeros((1), device=device)
            for i in range(ITERATION_TIME):
                train_out = model.forward(mini_train_input[i])
                train_loss = criterion(train_out, mini_train_label[i]) / ITERATION_TIME
                train_losses += train_loss
                train_loss.backward()         # Calculate gradient
            plt_loss.restore_loss_from_optimizer(train_losses, training=True)
            print("train loss", train_losses.item())
            return train_losses
        optimizer.step(closure)     # Closure because of Conjugate Gradient and LBFGS
#########################################################################
        # def closure():
        #     optimizer.zero_grad()           # Empty the gradient
        #     train_out = model.forward(train_input)
        #     train_loss = criterion(train_out, train_label)
        #     plt_loss.restore_loss_from_optimizer(train_loss, training=True)
        #     print("train loss", train_loss.item())
        #     train_loss.backward()         # Calculate gradient
        #     return train_loss
        # optimizer.step(closure)     # Closure because of Conjugate Gradient and LBFGS
#########################################################################

        """ Validation """
        model.eval()    # Enter evaluating mode
        with torch.no_grad():   # Turn off gradient computation
            val_out = model.forward(val_input).squeeze()
            val_loss = criterion(val_out, val_label)
            plt_loss.restore_loss_from_optimizer(val_loss, training=False)
            print("loss", val_loss.item())

        # """ Prediction """
        # with torch.no_grad():
        #     future = 1000
        #     pred = model.forward(val_input, future=future)
        #     loss = criterion(pred[:, :-future], val_label)    # Exclude predicted data
        #     print("test loss", loss.item())     # .item means get the int part from a tensor
        #     y = pred.detach().numpy()

        # """ Draw diagram """
        # plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # n = train_input.shape[1]    # 999
        # def draw(y_i, color):
        #     plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
        #     plt.plot(np.arange(n, n+future), y_i[n:], color + ":", linewidth=2.0)
        # draw(y[0], 'r')
        # draw(y[1], 'b')
        # draw(y[2], 'g')

        # plt.savefig("predict%d.pdf"%i)
        # plt.close()
        # # plt.grid()
        # # plt.show()

    # Plot the loss convergence
    plt_loss.plot_losses()
    