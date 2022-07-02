import os
import csv
import numpy as np

py_path = os.path.abspath(__file__)
py_dir = os.path.dirname(py_path)
acc = os.path.join(py_dir, 'accel-0.csv')
gyro = os.path.join(py_dir, 'gyro-0.csv')
mag = os.path.join(py_dir, 'mag-0.csv')
gps = os.path.join(py_dir, 'gps-0.csv')
def TXfilter(path, TXdata):
    #讀取檔案
    with open(path, newline = '') as csvfile:
        rows = csv.reader(csvfile, delimiter = ',')
        data = np.asarray(list(rows))
    #篩選出'TX', 'expired in Oct'，並存入TXdata中
    for i in range(len(data)):
        if (data[i][1] == 'TX     ') & (data[i][2] == '202110     '):
            for j in range(len(numlist)):
                new[j] = data[i][numlist[j]]
            print(path, data[i][1], data[i][2], TXdata.shape)
            TXdata = np.vstack((TXdata, new))
    return TXdata

acc_data = np.genfromtxt(self.data_src[0], delimiter=',', skip_header=1, max_rows=self.__max_row(0), dtype=np.float32)
gyro_data = np.genfromtxt(self.data_src[1], delimiter=',', skip_header=1, max_rows=self.__max_row(0), dtype=np.float32)
mag_data = np.genfromtxt(self.data_src[2], delimiter=',', skip_header=1, max_rows=self.__max_row(0), dtype=np.float32)
gps_data = np.genfromtxt(self.data_src[3], delimiter=',', skip_header=2, usecols=(9,10,11), max_rows=self.__max_row(1), dtype=np.float32)