import copy
import glob
import os
import time

import pandas as pd
from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from matplotlib import pyplot as plt
import datetime
from demo01.models import LiquidInfo, LifeTimeInfo, StateInfo, ThickInfo, WarmInfo, UserInfo, DateTest,ThickInfo2,ThickInfo3,ThickInfo4,LastId,LastId2,DatePath
import numpy as np
import random
import datetime
from stat import S_ISREG, ST_CTIME, ST_MODE
from nptdms import TdmsFile

def login(request):
    if request.method == "GET":
        return render(request, 'login.html')
    else:
        # 如果是POST请求，获取用户提交的数据
        print(request.POST)
        username = request.POST.get("user")
        print(username)
        password = request.POST.get("password")
        print(password)
        if username == 'root' and password == "123":
            return redirect('/info/ditch/')
        else:
            return render(request, 'login.html', {"error_msg": "用户名或密码错误"})

def ditchInfo(request):
    """
    需要传的值：1.厚度预警值
    :param request:
    :return:
    """
    data_warn = WarmInfo.objects.all().first()
    return render(request, 'ditchInfo.html', {"data_warn": data_warn})


def liquidInfo(request):
    data_warn = WarmInfo.objects.all().first()
    return render(request, 'liquidInfo.html', {"data_warn": data_warn})

def ironInfo(request):
    return render(request, 'ironInfo.html')

def warn(request):
    data = WarmInfo.objects.all().first()
    # print(data.thickness_warm, data.height_warm)
    return render(request, 'warn.html', {"data": data})

def warmEdit(request):
    nid = request.GET.get('nid')
    if request.method == "GET":
        # print('here',nid)
        data = WarmInfo.objects.filter(id=nid).first()
        return render(request, 'warmEdit.html', {"data": data})
    else:
        # print(request.POST.get('thickness'))
        # print(request.POST.get('height')
        thickness = request.POST.get("thickness")
        height = request.POST.get('height')
        WarmInfo.objects.filter(id=nid).update(thickness_warm=thickness,
                                               height_warm=height)
        return redirect('/info/warn/')

def history(request):
    return render(request, 'historyInfo.html',)

def historyInfo(request):
    dateset_thick = ThickInfo.objects.all()
    dates1 = []
    datas1 = []
    for item in dateset_thick:
        if item.date:
            result = item.date
            dates1.append(result)
            datas1.append(item.thickness)
    dateset_thick2 = ThickInfo2.objects.all()
    dates2 = []
    datas2 = []
    for item in dateset_thick2:
        if item.date:
            result = item.date
            dates2.append(result)
            datas2.append(item.thickness)
    dateset_thick3 = ThickInfo3.objects.all()
    dates3 = []
    datas3 = []
    for item in dateset_thick3:
        if item.date:
            result = item.date
            dates3.append(result)
            datas3.append(item.thickness)
    dateset_thick4 = ThickInfo4.objects.all()
    dates4 = []
    datas4 = []
    for item in dateset_thick4:
        if item.date:
            result = item.date
            dates4.append(result)
            datas4.append(item.thickness)
    data = {
        "datas_1": datas1,
        "dates_1": dates1,
        "datas_2": datas2,
        "dates_2": dates2,
        "datas_3": datas3,
        "dates_3": dates3,
        "datas_4": datas4,
        "dates_4": dates4,

    }
    return JsonResponse(data)

def historyInfoMain(request):
    dateset_thick = ThickInfo2.objects.all()
    dates1 = []
    datas1 = []
    for item in dateset_thick:
        if item.date:
            result = item.date
            dates1.append(result)
            datas1.append(item.thickness)
    data = {
        "datas_1": datas1[-700:-1],
        "dates_1": dates1[-700:-1],
    }
    return JsonResponse(data)

def userInfo(request):
    data = UserInfo.objects.all()
    print('user_data', data)
    return render(request, 'userInfo.html', {"data": data})

## open('./demo01/mymodel/parameters', "rb")

def getLifeDate(request):
    dataset = LifeTimeInfo.objects.all()
    # print("here1",dataset)
    index = len(dataset)
    result = dataset[index-1].lifeTime
    data_res = {
        "data": result,
    }
    # print("here",result)
    return JsonResponse(data_res)

def getStateData(request):
    dataset = StateInfo.objects.all()
    index = random.randint(0, len(dataset) - 1)
    result = dataset[index]
    data_res = {
        "systemType": result.systemType,
        'networkType': result.networkType,
        'runType': result.runType,
    }
    return JsonResponse(data_res)

def getThickData1(request):
    index1 = LastId.objects.get(id=1).lastid_1  # 201
    dataset = ThickInfo.objects.get(id=index1)
    result = dataset.thickness
    date = dataset.date.strftime("%Y-%m-%d %H:%M:%S")
    data = WarmInfo.objects.all().first()
    warn_thickness = data.thickness_warm
    state = 'green'
    if result < warn_thickness:
        state = 'red'
    data_res = {
        "thickness": result,
        "state": state,
        "date": date
    }
    return JsonResponse(data_res)

def getThickData1First(request):
    index1 = LastId.objects.get(id=1).lastid_1
    result = []
    date = []
    state = ''
    for i in range(1, 6):# 1 2 3 4 5
        index1 = index1 - i
        dataset = ThickInfo.objects.get(id=index1)
        result.append(dataset.thickness)
        date.append(dataset.date.strftime("%Y-%m-%d %H:%M:%S"))
        if i == 0:
            data = WarmInfo.objects.all().first()
            warn_thickness = data.thickness_warm
            state = 'green'
            if result[0] < warn_thickness:
                state = 'red'

    data_res = {
        "thickness_1": result[0],
        "date_1": date[0],
        "thickness_2": result[1],
        "date_2": date[1],
        "thickness_3": result[2],
        "date_3": date[2],
        "thickness_4": result[3],
        "date_4": date[3],
        "thickness_5": result[4],
        "date_5": date[4],
        "state": state,
    }
    # setThickData1()
    return JsonResponse(data_res)

def read_newtdmsfile(_path):#"/Users/yaoyaohao/Desktop/EMF数据/*.tdms"
# 获取目录中的 tdms 文件列表
    path = _path
    file_list = glob.glob(path+"/*.tdms")
    # 按照文件创建时间排序
    file_list.sort(key=lambda x: os.stat(x).st_ctime,reverse=True)
    file_path=file_list[0]
    file_info=os.stat(file_list[0])
    # 文件结束时间
    dt_end=datetime.datetime.fromtimestamp(file_info[ST_CTIME])
    # 文件名称
    file_name = os.path.basename(file_path)
    file_time, file_ext = os.path.splitext(file_name)
    #---------需要部分修改（开始采集时间和结束时间）

    # read a tdms file
    filenameS = file_path
    tdms_file = TdmsFile(filenameS)
    tdms_groups = tdms_file.groups()
    sheet_list = ["Events","Measured Data"]
    # Close the Pandas Excel writer and output the Excel file.

    df = tdms_file["Measured Data"].as_dataframe()
    df = pd.DataFrame(df)
    index_i=['timel','timeh']
    for i in range(1,61):
        index_i.append('CH '+str(i))
    df.columns=index_i
    _signal=[]
    j=0
    for i in (1,15,2,16):
       y=df['CH '+str(i)].tolist()
       _signal.append(y)
       j=j+1

    return _signal[0],_signal[1],_signal[2],_signal[3]

from demo01.mymodel import Bpnet
def setThickData1():
    step = 500
    beta = 0.001
    layer = [4, 32, 16, 8, 4, 2, 1]
    path = DatePath.objects.all().first().path
    x,a,b = Bpnet.load_data(path)
    # # data = [(np.array([[x_value[0]],[x_value[1]],[x_value[2]],[x_value[3]]]), np.array([y_value])) for x_value, y_value in zip(x, y)]
    model = Bpnet.BP(layer, Bpnet.tanh, Bpnet.tanh_derivative, Bpnet.loss_derivative)
    x = np.array(x)
    model.load_weights()
    # predict = model.predict(x)

    # todo 训练一个新模型 这里只预测一个数据 然后拿数据库的最后一个数据来做权重计算
    predict = model.predict(x[-51:-1])
    sum_finall = 0
    for item in predict:
        sum_finall = sum_finall+item

    predict_finall = sum_finall/50
    last_id = LastId2.objects.get(id=1).lastid_1
    last_data = ThickInfo.objects.get(id=last_id).thickness
    if predict_finall > (last_data/ 1000):
        result = (last_data / 1000) * 0.9987
    else:
        result = (last_data / 1000)* 0.998 + predict_finall*0.002
    result = result*1000
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ThickInfo.objects.create(thickness=result, date=date)
    index1 = LastId.objects.get(id=1).lastid_1  # 201
    index1_new = index1 + 1
    LastId.objects.filter(id=1).update(lastid_1=index1_new)
    a_rand = random.randint(-6,6)
    b_rand = random.randint(-6,6)
    a = result+a_rand
    b = result+b_rand
    ThickInfo2.objects.create(thickness=a, date=date)
    index2 = LastId.objects.get(id=1).lastid_2  # 201
    index2_new = index2 + 1
    LastId.objects.filter(id=1).update(lastid_2=index2_new)
    ThickInfo3.objects.create(thickness=b, date=date)
    index3 = LastId.objects.get(id=1).lastid_3  # 201
    index3_new = index3 + 1
    LastId.objects.filter(id=1).update(lastid_3=index3_new)
    d = (result+a+b)/3
    ThickInfo4.objects.create(thickness=d, date=date)
    index4 = LastId.objects.get(id=1).lastid_4  # 201
    index4_new = index4 + 1
    LastId.objects.filter(id=1).update(lastid_4=index4_new)
    LastId2.objects.filter(id=1).update(lastid_1=last_id + 1)
    life_state = ((result-550)/950)*100
    LifeTimeInfo.objects.create(lifeTime=life_state)

def getThickData2(request):
    index2 = LastId.objects.get(id=1).lastid_2  # 201
    dataset = ThickInfo2.objects.get(id=index2)
    result = dataset.thickness
    date = dataset.date.strftime("%Y-%m-%d %H:%M:%S")
    data = WarmInfo.objects.all().first()
    warn_thickness = data.thickness_warm
    state = 'green'
    if result < warn_thickness:
        state = 'red'

    data_res = {
        "thickness": result,
        "state": state,
        "date": date
    }
    return JsonResponse(data_res)

def getThickData2First(request):
    index2 = LastId.objects.get(id=1).lastid_2  # 201
    result = []
    date = []
    state = ''
    for i in range(1,6): # 0 1 2 3 4
        index2 = index2 - i
        dataset = ThickInfo2.objects.get(id=index2)
        result.append(dataset.thickness)
        date.append(dataset.date.strftime("%Y-%m-%d %H:%M:%S"))
        # 最后一个状态
        if i == 0:
            data = WarmInfo.objects.all().first()
            warn_thickness = data.thickness_warm
            state = 'green'
            if result[0] < warn_thickness:
                state = 'red'

    data_res = {
        "thickness_1": result[0],
        "date_1": date[0],
        "thickness_2": result[1],
        "date_2": date[1],
        "thickness_3": result[2],
        "date_3": date[2],
        "thickness_4": result[3],
        "date_4": date[3],
        "thickness_5": result[4],
        "date_5": date[4],
        "state": state,
    }
    # setThickData1()
    return JsonResponse(data_res)

def getThickData3(request):
    index3 = LastId.objects.get(id=1).lastid_3 # 201
    dataset = ThickInfo3.objects.get(id=index3)
    result = dataset.thickness
    date = dataset.date.strftime("%Y-%m-%d %H:%M:%S")
    data = WarmInfo.objects.all().first()
    warn_thickness = data.thickness_warm
    state = 'green'
    if result < warn_thickness:
        state = 'red'
    data_res = {
        "thickness": result,
        "state": state,
        "date": date
    }
    return JsonResponse(data_res)

def getThickData3First(request):
    index3 = LastId.objects.get(id=1).lastid_3  # 201
    result = []
    date = []
    state = ''
    for i in range(1, 6): # 0 1 2 3 4
        index3 = index3 - i
        dataset = ThickInfo3.objects.get(id=index3)
        result.append(dataset.thickness)
        date.append(dataset.date.strftime("%Y-%m-%d %H:%M:%S"))
        # 最后一个状态
        if i == 0:
            data = WarmInfo.objects.all().first()
            warn_thickness = data.thickness_warm
            state = 'green'
            if result[0] < warn_thickness:
                state = 'red'

    data_res = {
        "thickness_1": result[0],
        "date_1": date[0],
        "thickness_2": result[1],
        "date_2": date[1],
        "thickness_3": result[2],
        "date_3": date[2],
        "thickness_4": result[3],
        "date_4": date[3],
        "thickness_5": result[4],
        "date_5": date[4],
        "state": state,
    }
    # setThickData1()
    return JsonResponse(data_res)

def getThickData4(request):
    index4 = LastId.objects.get(id=1).lastid_4  # 201
    dataset = ThickInfo4.objects.get(id=index4)
    result = dataset.thickness
    date = dataset.date.strftime("%Y-%m-%d %H:%M:%S")
    data = WarmInfo.objects.all().first()
    warn_thickness = data.thickness_warm
    state = 'green'
    if result < warn_thickness:
        state = 'red'
    data_res = {
        "thickness": result,
        "state": state,
        "date": date
    }
    return JsonResponse(data_res)

def getThickData4First(request):
    index4 = LastId.objects.get(id=1).lastid_4  # 201
    result = []
    date = []
    state = ''
    for i in range(1, 6):# 0 1 2 3 4
        index4 = index4 - i
        dataset = ThickInfo4.objects.get(id=index4)
        result.append(dataset.thickness)
        date.append(dataset.date.strftime("%Y-%m-%d %H:%M:%S"))
        # 最后一个状态
        if i == 0:
            data = WarmInfo.objects.all().first()
            warn_thickness = data.thickness_warm
            state = 'green'
            if result[0] < warn_thickness:
                state = 'red'

    data_res = {
        "thickness_1": result[0],
        "date_1": date[0],
        "thickness_2": result[1],
        "date_2": date[1],
        "thickness_3": result[2],
        "date_3": date[2],
        "thickness_4": result[3],
        "date_4": date[3],
        "thickness_5": result[4],
        "date_5": date[4],
        "state": state,
    }
    # setThickData1()
    return JsonResponse(data_res)

def getLiquidData(request):
    dataset = LiquidInfo.objects.all()
    index = random.randint(0, len(dataset) - 1)
    result = dataset[index].li_height
    # data = WarmInfo.objects.all().first()
    # warn_thickness = data.thickness_warm
    # state = 'green'
    # if result < warn_thickness:
    #     state = 'red'
    data_res = {
        "height": result,
        # "state": state,
    }
    return JsonResponse(data_res)

def test(request):
    # data = ['2022/01/01', '2022/01/01', '2022/01/01']
    print("发送了设置数据的请求")
    print(os.getcwd())
    setThickData1()
    return JsonResponse({'state':'ok'})




