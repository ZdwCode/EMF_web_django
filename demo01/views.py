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

# Create your views here.
# def index(request):
#     return render(request, 'index.html')


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


# def default(request):
#     return render(request, 'layout_default.html')


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


def historyInfo(request):
    if request.method == "GET":
        return render(request, 'historyInfo.html')
    else:
        start_time = request.POST.get("start")
        end_time = request.POST.get("end")
        if start_time == '':
            return render(request, 'historyInfo.html')
        # start_time = datetime.date(int(start_time.split('-')[0]),
        #                            int(start_time.split('-')[1]),
        #                            int(start_time.split('-')[2]))

        # 判断日期的大小
        # print(type(start_time - end_time))
        # print(type((start_time - end_time).days))
        dateset_thick = ThickInfo.objects.all()
        dates = []
        datas = []
        for item in dateset_thick:
            result = str(item.date)
            if start_time in result:
                result = str(result).replace('-', '/')
                myresult = result.split(' ')[1].split('+')[0]
                dates.append(myresult)
                datas.append(item.thickness)
        data = {
            "datas": datas,
            "dates": dates,
        }
        return render(request, 'historyInfo.html', {"data": data})


def userInfo(request):
    data = UserInfo.objects.all()
    print('user_data', data)
    return render(request, 'userInfo.html', {"data": data})


## open('./demo01/mymodel/parameters', "rb")
def datatest(request):

    # 添加液面高度数据 40-180 随机生成100组
    # dates = np.random.randint(40, 180, 100)
    # for data in dates:
    #     LiquidInfo.objects.create(li_height=data)

    # 添加系统状态数据 40-100 随机生成60组
    # dates_sys = np.random.randint(0, 2, 50)
    # dates_net = np.random.randint(0, 2, 50)
    # dates_run = np.random.randint(0, 2, 50)
    # for i in range(50):
    #     StateInfo.objects.create(systemType=dates_sys[i],
    #                              networkType=dates_net[i],
    #                              runType=dates_run[i])

    # 添加厚度数据

    df = pd.read_csv('./demo01/mymodel/predict_data.csv')
    datas_thick = df['predict_data']
    for data in datas_thick:
        ThickInfo.objects.create(thickness=data)
        ThickInfo2.objects.create(thickness=data)
        ThickInfo3.objects.create(thickness=data)
        ThickInfo4.objects.create(thickness=data)

    # 测试日期数据
    # date = '2022-02-01'
    # DateTest.objects.create(date=date)
    # dataset_li = LiquidInfo.objects.all()
    # oneDay = datetime.timedelta(days=1)
    # for index in range(1,len(dataset_li)):
    #     dataset_li[index].date = dataset_li[index-1].date + oneDay
    #     LiquidInfo.objects.filter(id=index).update(date=dataset_li[index].date)
    # # print(temp.replace('-', ','))

    # 测试日期数据
    # dataset_thick = ThickInfo.objects.all()
    # oneDay = datetime.timedelta(days=1)
    # for index in range(1,len(dataset_thick)):
    #     dataset_thick[index].date = dataset_thick[index-1].date + oneDay
    #     ThickInfo.objects.filter(id=index).update(date=dataset_thick[index].date)
    #     # print(temp.replace('-', ','))
    return HttpResponse("添加成功")


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
    print('厚度1 取到的id:', index1)
    index1_new = index1 + 1
    print('厚度1 增加后的id:', index1_new)
    LastId.objects.filter(id=1).update(lastid_1=index1_new)
    # index = 5978
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
    layer = [4, 5, 6, 5, 4, 1]
    # 从na'li
    path = DatePath.objects.all().first().path
    x,a,b = Bpnet.load_data(path)
    # # data = [(np.array([[x_value[0]],[x_value[1]],[x_value[2]],[x_value[3]]]), np.array([y_value])) for x_value, y_value in zip(x, y)]
    model = Bpnet.BP(layer, Bpnet.tanh, Bpnet.tanh_derivative, Bpnet.loss_derivative)
    x = np.array(x)
    model.load_weights()
    print('参数加载完毕')
    # predict = model.predict(x)

    # todo 训练一个新模型 这里只预测一个数据 然后拿数据库的最后一个数据来做权重计算
    predict = model.predict(x[-51:-1])
    sum_finall = 0
    for item in predict:
        sum_finall = sum_finall+item

    predict_finall = sum_finall/50
    last_id = LastId2.objects.get(id=1).lastid_1
    last_data = ThickInfo.objects.get(id=last_id).thickness
    result = (last_data/1000)*0.98 + predict_finall*0.02
    result = result*1000
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ThickInfo.objects.create(thickness=result, date=date)
    ThickInfo2.objects.create(thickness=a, date=date)
    ThickInfo3.objects.create(thickness=b, date=date)
    d = (result+a+b)/3
    ThickInfo4.objects.create(thickness=d, date=date)
    LastId2.objects.filter(id=1).update(lastid_1=last_id + 1)
    life_state = (result/950)*100
    LifeTimeInfo.objects.create(lifetime=life_state)


def getThickData2(request):
    index2 = LastId.objects.get(id=1).lastid_2  # 201
    print('厚度2 取到的id:', index2)
    index2_new = index2 + 1
    print('厚度2 增加后的id:', index2_new)
    LastId.objects.filter(id=1).update(lastid_2=index2_new)
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


def getThickData3(request):
    index3 = LastId.objects.get(id=1).lastid_3 # 201
    print('厚度3 取到的id:', index3)
    index3_new = index3 + 1
    print('厚度3 增加后的id:', index3_new)
    LastId.objects.filter(id=1).update(lastid_3=index3_new)

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


def getThickData4(request):
    index4 = LastId.objects.get(id=1).lastid_4  # 201
    print('厚度4 取到的id:', index4)
    index4_new = index4 + 1
    print('厚度4 增加后的id:', index4_new)
    LastId.objects.filter(id=1).update(lastid_4=index4_new)
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


# 修改数据库的时间
def changedate(request):
    in_date = '2022-10-31 00:00:00'
    dt = datetime.datetime.strptime(in_date, "%Y-%m-%d %H:%M:%S")
    # LastId.objects.filter(id=1).update(lastid_3=index3 + 1)
    for item1 in range(5328,5796):
        new_date = (dt + datetime.timedelta(hours=(item1-5328+1))).strftime("%Y-%m-%d %H:%M:%S")
        ThickInfo.objects.filter(id=item1).update(date=new_date)
    for item2 in range(201,850):
        new_date = (dt + datetime.timedelta(hours=(item2 - 201 + 1))).strftime("%Y-%m-%d %H:%M:%S")
        ThickInfo2.objects.filter(id=item2).update(date=new_date)
    for item3 in range(201,850):
        new_date = (dt + datetime.timedelta(hours=(item3 - 201 + 1))).strftime("%Y-%m-%d %H:%M:%S")
        ThickInfo3.objects.filter(id=item3).update(date=new_date)
    for item4 in range(201,850):
        new_date = (dt + datetime.timedelta(hours=(item4 - 201 + 1))).strftime("%Y-%m-%d %H:%M:%S")
        ThickInfo4.objects.filter(id=item4).update(date=new_date)

    return HttpResponse('ok')
# if __name__ == '__main__':
#     caiji_houdu=950
#     #选择简化后的数据（每分钟的）和未简化的数据（每秒10次）
#     sig1, sig15, sig2, sig16 = read_newtdmsfile("/Users/yaoyaohao/Desktop/EMF数据")
#     a=calculate1(sig1,sig15,sig2,sig16,caiji_houdu)
#     b=calculate2(sig1,sig15,caiji_houdu)
#     print(b)



