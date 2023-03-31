# python-Error Back Propagation
# coding=utf-8
import glob
import os
import pickle
import openpyxl
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
from stat import S_ISREG, ST_CTIME, ST_MODE
from nptdms import TdmsFile
import copy
import pandas

def loss_derivative(output_activations, y):
    return 2 * (output_activations - y)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) * np.tanh(z)


def mean_squared_error(predictY, realY):
    Y = numpy.array(realY)
    return np.sum((predictY - Y) ** 2) / realY.shape[0]


class BP:
    def __init__(self, sizes, activity, activity_derivative, loss_derivative):
        # [4, 32, 64, 128, 32, 1]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((nueron, 1)) for nueron in sizes[1:]]
        self.weights = [np.random.randn(next_layer_nueron, nueron) for nueron, next_layer_nueron in
                        zip(sizes[:-1], sizes[1:])]
        self.activity = activity
        self.activity_derivative = activity_derivative
        self.loss_derivative = loss_derivative

    def predict(self, a):
        re = a.T
        n = len(self.biases) - 1
        for i in range(n):
            b, w = self.biases[i], self.weights[i]
            re = self.activity(np.dot(w, re) + b)
        re = np.dot(self.weights[n], re) + self.biases[n]
        return re.T

    def update_batch(self, batch, learning_rate):
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_temp_b, delta_temp_w = self.update_parameter(x, y)
            temp_w = [w + dw for w, dw in zip(temp_w, delta_temp_w)]
            temp_b = [b + db for b, db in zip(temp_b, delta_temp_b)]
        # print(self.weights)
        # print([sw - (learning_rate / len(batch)) * w for sw, w in zip(self.weights, temp_w)])
        self.weights = [sw - (learning_rate / len(batch)) * w for sw, w in zip(self.weights, temp_w)]
        self.biases = [sb - (learning_rate / len(batch)) * b for sb, b in zip(self.biases, temp_b)]
        # print('更新后')
        # print(self.weights)

    def update_parameter(self, x, y):
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        n = len(self.biases)
        for i in range(n):
            b, w = self.biases[i], self.weights[i]
            z = np.dot(w, activation) + b
            zs.append(z)
            if i != n - 1:
                # 激活层
                activation = self.activity(z)
            else:
                activation = z
            activations.append(activation)
        # 计算最后一层的损失
        d = self.loss_derivative(activations[-1], y)
        temp_b[-1] = d
        temp_w[-1] = np.dot(d, activations[-2].T)
        for i in range(2, self.num_layers):
            z = zs[-i]
            d = np.dot(self.weights[-i + 1].T, d) * self.activity_derivative(z)
            temp_b[-i] = d
            temp_w[-i] = np.dot(d, activations[-i - 1].T)
        return (temp_b, temp_w)

    def fit(self, train_data, epochs, batch_size, learning_rate, validation_data=None):
        n = len(train_data)
        losses = []
        for j in range(epochs):
            np.random.shuffle(train_data)
            # 分批 一批次一批次的训练
            batches = [train_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if validation_data != None:
                val_pre = self.predict(validation_data[0])
                loss = mean_squared_error(val_pre, validation_data[1])
                print("Epoch", j + 1, '/', epochs,
                      '  val loss:%12.12f' % loss)
                losses.append(loss)
        return epochs, losses
    def save_weights(self):
        paremeters = {
            "biases": self.biases,
            "weights": self.weights,
        }
        with open('parameters_new_1', "wb") as f:
            pickle.dump(paremeters, f)

    def load_weights(self):
        # 服务器地址
        with open('.\django\contrib\\admin\static\model\parameters_new', "rb") as f:
        # with open('.\demo01\mymodel\parameters_new', "rb") as f:
            weights = pickle.load(f)
            self.biases = weights["biases"]
            self.weights = weights["weights"]
        #  本地地址
        # with open('./parameters', "rb") as f:
        #     weights = pickle.load(f)
        #     self.biases = weights["biases"]
        #     self.weights = weights["weights"]


def removeData(data):
    # 去除冗余数据
    y0 = data[0]
    y1 = data[1]
    y2 = data[2]
    y3 = data[3]
    y0_simple = []
    y1_simple = []
    y2_simple = []
    y3_simple = []
    result = []
    for i in range(len(y0)):
        if i % 400 == 0:
            y0_simple.append(y0[i])
            y1_simple.append(y1[i])
            y2_simple.append(y2[i])
            y3_simple.append(y3[i])
    # titanic.to_excel("../data2/EMF_11.21_2_simple_used.xlsx",index_label=False,index=False)
    result.append(y0_simple)
    result.append(y1_simple)
    result.append(y2_simple)
    result.append(y3_simple)
    return result


def smooth(data):
    y0 = data[0]
    y1 = data[1]
    y2 = data[2]
    y3 = data[3]
    y0_smooth = []
    y1_smooth = []
    y2_smooth = []
    y3_smooth = []
    result = []
    k = 0
    winds = 3000
    for i in range(winds, len(y0)):
        total0 = 0
        total1 = 0
        total2 = 0
        total3 = 0
        if i == winds:
            for j in range(0, winds):  # 0-999
                total0 += y0[i - j]
                total1 += y1[i - j]
                total2 += y2[i - j]
                total3 += y3[i - j]
            y0_smooth.append(total0 / winds)
            y1_smooth.append(total1 / winds)
            y2_smooth.append(total2 / winds)
            y3_smooth.append(total3 / winds)
            k += 1
        else:
            total0 = y0_smooth[k - 1] * winds - y0[i - winds] + y0[i]
            y0_smooth.append(total0 / winds)
            total1 = y1_smooth[k - 1] * winds - y1[i - winds] + y1[i]
            y1_smooth.append(total1 / winds)
            total2 = y2_smooth[k - 1] * winds - y2[i - winds] + y2[i]
            y2_smooth.append(total2 / winds)
            total3 = y3_smooth[k - 1] * winds - y3[i - winds] + y3[i]
            y3_smooth.append(total3 / winds)
            k += 1
    result.append(y0_smooth)
    result.append(y1_smooth)
    result.append(y2_smooth)
    result.append(y3_smooth)
    return result


def normalization(data):
    result = []
    u0 = data[0]
    u1 = data[1]
    u2 = data[2]
    u3 = data[3]
    u0_max = max(u0)
    u0_min = min(u0)
    u0_under = u0_max - u0_min

    u1_max = max(u1)
    u1_min = min(u1)
    u1_under = u1_max - u1_min

    u2_max = max(u2)
    u2_min = min(u2)
    u2_under = u2_max - u2_min

    u3_max = max(u3)
    u3_min = min(u3)
    u3_under = u3_max - u3_min

    # 归一化
    for i in range(len(u1)):
        u0[i] = (u0[i] - u0_min) / u0_under
        u1[i] = (u1[i] - u1_min) / u1_under
        u2[i] = (u2[i] - u2_min) / u2_under
        u3[i] = (u3[i] - u3_min) / u3_under
    result.append(u0)
    result.append(u1)
    result.append(u2)
    result.append(u3)
    return result


def readsimple(i,_url):#直接读取数据简化后的文件
    EMF=[]
    wb = openpyxl.load_workbook(r''+_url+str(i)+'.xlsx')
    ws = wb['Sheet1']
    for row in ws.rows:
        for cell in row:
            EMF.append(cell.value)
    return EMF

def readfoursimple(_a,_b,_c,_d,_url):
    a=readsimple(_a,_url)
    b=readsimple(_b,_url)
    c=readsimple(_c,_url)
    d=readsimple(_d,_url)
    return a,b,c,d
def readallsimple(_url):
    a=[]
    for i in [1,15,2,16, 3,17,4,18, 5,21,6,22, 7,23,8,24, 11,25,12,26, 13,27,14,28]:
      a.append(readsimple(i, _url))
    return a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],a[18],a[19],a[20],a[21],a[22],a[23]

def load_new_data():
    url = r'G:\WeChat Files\WeChat Files\wxid_62au669s5k7c22\FileStorage\File\2023-03\11.21-12.22\11.21-12.22.'
    sig1, sig15, sig2, sig16 = readfoursimple(1, 15, 2, 16, url)
    sig1=sig1[4000:13490]
    sig15=sig15[4000:13490]
    sig2=sig2[4000:13490]
    sig16=sig16[4000:13490]
    return sig1, sig15, sig2, sig16


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bisect
from scipy import signal
import openpyxl
from nptdms import TdmsFile
from nptdms import tdms
import glob
import os
import datetime
from stat import S_ISREG, ST_CTIME, ST_MODE
def EMFsimple(_y):
    j=0
    list1=_y
    list2=[]
    for i in list1:
        if j%600==0 :# 设置10为按秒出结果，600为按分出以此类推
            list2.append(list1[j])
        j=j+1
    return list2

# def read_newtdmsfile(_path):#"/Users/yaoyaohao/Desktop/EMF数据/*.tdms"
# # 获取目录中的 tdms 文件列表
#     path=_path
#     file_list = glob.glob(path+"/*.tdms")
#
#     # 按照文件创建时间排序
#     file_list.sort(key=lambda x: os.stat(x).st_ctime,reverse=True)
#     file_path=file_list[0]
#     file_info=os.stat(file_list[0])
#     # 文件结束时间
#     dt_end=datetime.datetime.fromtimestamp(file_info[ST_CTIME])
#     # 文件名称
#     file_name = os.path.basename(file_path)
#     file_time, file_ext = os.path.splitext(file_name)
#     #---------需要部分修改（开始采集时间和结束时间）
#
#     # read a tdms file
#     filenameS = file_path
#     tdms_file = TdmsFile(filenameS)
#     tdms_groups = tdms_file.groups()
#     sheet_list = ["Events","Measured Data"]
#     # Close the Pandas Excel writer and output the Excel file.
#
#     df = tdms_file["Measured Data"].as_dataframe()
#     df=pd.DataFrame(df)
#     index_i=['timel','timeh']
#     for i in range(1,61):
#         index_i.append('CH '+str(i))
#     df.columns=index_i
#     _signal=[]
#     j=0
#     for i in (1,15,2,16):
#        y=df['CH '+str(i)].tolist()
#        _signal.append(EMFsimple(y))
#        j=j+1
#
#     return _signal[0],_signal[1],_signal[2],_signal[3]

#--------------处理部分-------------#
#---------------------------#
def trend(_signal):
    x = [i for i in range(len(_signal))]
    param = np.polyfit(x, _signal, 4)  # 用2次多项式拟合x，y数组,返回多项式系数
    y_poly_func = np.poly1d(param)  # 拟合完之后，用生成的多项式系数用来生成多项式函数
    y_poly = y_poly_func(x)
    return y_poly
def medianSlidingWindow(nums, k):
    n = len(nums)
    window = []
    ans = []

    for i in range(n):
        pos = bisect.bisect_left(window, nums[i])
        window[pos: pos] = [nums[i]]

        if len(window) > k:
            a = nums[i - k]
            pos = bisect.bisect_left(window, a)
            window[pos: pos + 1] = []

        if len(window) == k:
            num = (window[k // 2] + window[(k - 1) // 2]) / 2
            ans.append(num)

    return ans
def lowpass(_signal):
    b, a = signal.butter(3, 0.003, 'lowpass')  # 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a,_signal)
    return filtedData
#
def calculate1(_signal1,_signal2,_signal3,_signal4,init_data):
    signal1=_signal1
    signal2=_signal2
    signal3=_signal3
    signal4=_signal4
    c=init_data#起始厚度

    _test=[]
    e = []
    for window in range(0, len(signal1), 60):
        # print(window)
        Mean_all = []
        b = window + 1
        y_mean_signla1 = np.mean(signal1[window:b])
        Mean_all.append((y_mean_signla1))

        y_mean_signla2=np.mean(signal2[window:b])
        Mean_all.append((y_mean_signla2))

        y_mean_signla3 = np.mean(signal3[window:b])
        Mean_all.append((y_mean_signla3))

        y_mean_signla4 = np.mean(signal4[window:b])
        Mean_all.append((y_mean_signla4))

        U_100 = Mean_all[2]
        U_200 = Mean_all[3]
        U_100_p = Mean_all[0]
        U_200_p = Mean_all[1]
        a = U_100 + U_200
        b = U_100_p + U_200_p
        d = 0.0087 / abs(a / b)
        e.append(d)
        xmin = np.min(e)
        xmax = np.max(e)
        y = np.interp(e, (xmin, xmax), (0, 1))
        test = (c - 1 / 100 * 5 * y[0] * abs(a / b))
        c = test
        # print(e[0] * abs(a / b))
        # ------------------#
        _test.append(test)
        # print(_test[-1])

    return _test[-1]

def calculate2(_signal1,_signal2,init_data):
    diff=[]
    der=[]
    test = []
    z=[]
    signal1=_signal1
    signal2=_signal2
    c=init_data
    for i in range(len(signal1)):
        diff.append(signal2[i]-signal1[i])
    plt.plot(diff)
    for i in range(len(signal1)):
        der.append(1/diff[i])
    plt.plot(der)
    y_pinghua = medianSlidingWindow(der, 60)
    y_pinghua_low=lowpass(y_pinghua)
    y_pinghua_low_trend=trend(y_pinghua_low)
    x = [i for i in range(len(y_pinghua_low_trend))]
    c = init_data
    coeff=0.2
    for i in range(len(y_pinghua_low_trend)):
        z.append(y_pinghua_low_trend[0] - y_pinghua_low_trend[i])#用一个新的来储存，便于后面限定范围
    xmin = np.min(z)
    xmax = np.max(z)
    y = np.interp(z, (xmin, xmax), (0, 0.01))
    for i in range(len(y_pinghua_low_trend)):
        x = c - coeff * y[-1]#
        c = x
        test.append(x)
    return test[-1]

def savesignal(_result,_url):
    df = pd.DataFrame(_result)
    df.to_excel(_url, index=False, header=False)


def load_data(path):
    """
    :param data: 处理完毕后的数据
    :return:
    """
    X = []
    Y = []  #
    data = []
    # 服务器地址
    data0, data1, data2, data3 = read_newtdmsfile(path)
    # todo 姚灏的方法加载这 然后写进数据库
    caiji_houdu = 900
    a=calculate1(data0, data1, data2, data3,caiji_houdu)
    b=calculate2(data0, data1,caiji_houdu)
    # 本地测试地址
    # data0, data1, data2, data3 = read_newtdmsfile('../static/datas/')
    #data0, data1, data2, data3 = load_new_data()
    data.append(data0)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    # 平滑
    result1 = smooth(data)
    # 去冗余
    result2 = removeData(result1)
    # 归一化
    result3 = normalization(result2)
    # result3 = result2
    u0 = result3[0]
    u1 = result3[1]
    u2 = result3[2]
    u3 = result3[3]
    start = 0.900
    for i in range(len(u0)):
        x = []
        y = []
        x.append(u0[i])
        x.append(u1[i])
        x.append(u2[i])
        x.append(u3[i])
        y.append(start)
        X.append(x)
        Y.append(y)
        start -= 0.00000072
    return X, a, b


def read_newtdmsfile(path):  # "/Users/yaoyaohao/Desktop/EMF数据/*.tdms"
    # 获取目录中的 tdms 文件列表
    # path=_path
    # file_list = glob.glob(path+"/*.tdms")
    # # 按照文件创建时间排序
    # file_list.sort(key=lambda x: os.stat(x).st_ctime,reverse=True)
    # file_path = file_list[0]
    # file_info = os.stat(file_list[0])
    # # 文件结束时间
    # dt_end=datetime.datetime.fromtimestamp(file_info[ST_CTIME])
    # # 文件名称
    # file_name = os.path.basename(file_path)
    # file_time, file_ext = os.path.splitext(file_name)
    # #---------需要部分修改（开始采集时间和结束时间）
    # G:\poststudy\2023-03\dist\dist\manage\django\contrib\admin\static\datas
    file_path = path
    # file_path = 'E:\pythonProject\laigang_web 2\demo01\static\datas\\数据_13日10时43分_7#.tdms'
    #file_path = '.\demo01\static\datas\\数据_13日10时43分_7#.tdms'
    # read a tdms file
    filenameS = file_path
    print('读到的文件路径为:', path)
    tdms_file = TdmsFile(filenameS)
    tdms_groups = tdms_file.groups()
    sheet_list = ["Events", "Measured Data"]
    # Close the Pandas Excel writer and output the Excel file.

    df = tdms_file["Measured Data"].as_dataframe()
    df = pd.DataFrame(df)
    # index_i = []
    index_i = ['timel','timeh'] # 22.1121日14时29分_7# .tdms  22.1123日23时39分_11#.tdms 数据_07日12时17分_13#.tdms 额外加
    for i in range(1, 61):
        index_i.append('CH ' + str(i)) # 11月数据_28日14时46分_12#.tdms 22.1121日14时29分_7# .tdms 22.1123日23时39分_11#.tdms 数据_07日12时17分_13#.tdms
        # index_i.append('3-' + str(i)) # 1_22年11月14日16时23分_0#.tdms
    df.columns = index_i
    _signal = []
    j =  0
    for i in (1, 15, 2, 16):
        y = df['CH ' + str(i)].tolist() # 11月数据_28日14时46分_12#.tdms 22.1121日14时29分_7# .tdms
        # y = df['3-' + str(i)].tolist()# 1_22年11月14日16时23分_0#.tdms
        _signal.append(y)
        j = j + 1

    return _signal[0], _signal[1], _signal[2], _signal[3]


if __name__ == "__main__":
    # 14 21 23 28
    step = 500
    beta = 0.001
    layer = [4, 5, 6, 5, 4, 1]
    #layer = [4, 5, 3, 1]
    x, y, = load_data()
    data = [(np.array([[x_value[0]], [x_value[1]], [x_value[2]], [x_value[3]]]), np.array([y_value])) for
            x_value, y_value in zip(x, y)]
    model = BP(layer, tanh, tanh_derivative, loss_derivative)
    x = np.array(x)
    # y = np.array(y)
    epochs, losses = model.fit(train_data=data, epochs=8000, batch_size=64, learning_rate=beta, validation_data=(x, y))
    #model.load_weights()
    model.save_weights()
    predict = model.predict(x)
    predict_ope = copy.deepcopy(predict)
    predict_ope[0] = 0.900
    for i in range(1, len(predict)):
        if abs(predict_ope[i] - predict_ope[i - 1]) < 0.04:
            predict_ope[i] = predict_ope[i - 1] * 0.98 + predict_ope[i] * 0.02
        else:
            predict_ope[i] = predict_ope[i - 1] * 0.9998 + predict_ope[i] * 0.0002
    x1 = range(len(predict))
    # # print(predict)
    plt.plot(x1, predict, "-r", linewidth=2, label='true')
    plt.plot(x1, predict_ope, "-b", linewidth=1, label='predict_1')
    plt.plot(x1, y, "-g", linewidth=1, label='predict_2')

    results_1 = []
    for i in predict_ope:
        results_1.append(i[0]*1000)
    plt.legend()
    plt.show()
    df = pandas.DataFrame({
        'predict_data': results_1[:]
    })
    df.to_csv('predict_data.csv')

    results_1 = []
    for i in y:
        results_1.append(i[0] * 1000)
    plt.legend()
    plt.show()
    df = pandas.DataFrame({
        'true_data': results_1[:]
    })
    df.to_csv('true_data.csv')

    results_1 = []
    for i in predict:
        results_1.append(i[0] * 1000)
    plt.legend()
    plt.show()
    df = pandas.DataFrame({
        'predict_original_data': results_1[:]
    })
    df.to_csv('predict_original_data.csv')
    #x2 = range(epochs)
    #plt.plot(x2, losses, "-r", linewidth=2, label='origin')
    #plt.show()
