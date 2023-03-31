import datetime
import time

import openpyxl


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
if __name__ == "__main__":
    # 选择简化后的数据（每分钟的）和未简化的数据（每秒10次）
    # url = "/Users/yaoyaohao/Desktop/EMF/tdms转化/EMFsimple/11.21-12.22/11.21-12.22."  # 11.21-12.22/11.21-12.22.
    url = r'G:\WeChat Files\WeChat Files\wxid_62au669s5k7c22\FileStorage\File\2023-03\11.21-12.22\11.21-12.22.'
    # 9.25.9 #11.21.14 #11.23.23
    # 11.28.14/EMFsimple_11.28.#12.7.12 #12.22.13 #
    # 2023.1.19/EMFsimple_2023.1.19.
    # /12.22-2.2/12.22-2.2.
    # url="/Users/yaoyaohao/Desktop/EMF/tdms转化/EMForiginal/11.23.23/EMF_11.23."#9.25.9 #11.21.14 #11.23.23
    # 11.28.14 #12.7.12 #12.22.13
    # 计算厚度
    sig1, sig15, sig2, sig16 = readfoursimple(1, 15, 2, 16, url)
    # (1,15,2,16, 3,17,4,18, 5,21,6,22, 7,23,8,24, 11,25,12,26, 13,27,14,28)

    sig1, sig15, sig2, sig16, \
    sig3, sig17, sig4, sig18, \
    sig5, sig21, sig6, sig22, \
    sig7, sig23, sig8, sig24, \
    sig11, sig25, sig12, sig26, \
    sig13, sig27, sig14, sig28 = readallsimple(url)
    # 选取小修前的部分配合11.21-12.22/11.21-12.22.使用
    sig1 = sig1[0:13490]
    sig15 = sig15[0:13490]
    sig2 = sig2[0:13490]
    sig16 = sig16[0:13490]
    sig3 = sig3[0:13490]
    sig17 = sig17[0:13490]
    sig4 = sig4[0:13490]
    sig18 = sig18[0:13490]
    sig5 = sig5[0:13490]
    sig21 = sig21[0:13490]
    sig6 = sig6[0:13490]
    sig22 = sig22[0:13490]
    sig7 = sig7[0:13490]
    sig23 = sig23[0:13490]
    sig8 = sig8[0:13490]
    sig24 = sig24[0:13490]
    sig11 = sig11[0:13490]
    sig25 = sig25[0:13490]
    sig12 = sig12[0:13490]
    sig26 = sig26[0:13490]
    sig13 = sig13[0:13490]
    sig27 = sig27[0:13490]
    sig14 = sig14[0:13490]
    sig26 = sig26[0:13490]

    # sig1=sig1[4000:13490]
    # sig15=sig15[4000:13490]
    # sig2=sig2[4000:13490]
    # sig16=sig16[4000:13490]
    # sig3 = sig3[4000:13490]
    # sig17 = sig17[4000:13490]
    # sig4= sig4[4000:13490]
    # sig18 = sig18[4000:13490]
    # sig5 = sig5[4000:13490]
    # sig21 = sig21[4000:13490]
    # sig6= sig6[4000:13490]
    # sig22 = sig22[4000:13490]
    # sig7 = sig7[4000:13490]
    # sig23 = sig23[4000:13490]
    # sig8 = sig8[4000:13490]
    # sig24 = sig24[4000:13490]
    # sig11 = sig11[4000:13490]
    # sig25 = sig25[4000:13490]
    # sig12= sig12[4000:13490]
    # sig26 = sig26[4000:13490]
    # sig13 = sig13[4000:13490]
    # sig27 = sig27[4000:13490]
    # sig14 = sig14[4000:13490]
    # sig26 = sig26[4000:13490]

    # 选取小修后的部分配合11.21-12.22/11.21-12.22.使用
    # sig1 = sig1[15651:]
    # sig15 = sig15[15651:]
    # sig2 = sig2[15651:]
    # sig16 = sig16[15651:]
    # sig3 = sig3[15651:]
    # sig17 = sig17[15651:]
    # sig4 = sig4[15651:]
    # sig18 = sig18[15651:]
    # sig5 = sig5[15651:]
    # sig21 = sig21[15651:]
    # sig6 = sig6[15651:]
    # sig22 = sig22[15651:]
    # sig7 = sig7[15651:]
    # sig23 = sig23[15651:]
    # sig8 = sig8[15651:]
    # sig24 = sig24[15651:]
    # sig11 = sig11[15651:]
    # sig25 = sig25[15651:]
    # sig12 = sig12[15651:]
    # sig26 = sig26[15651:]
    # sig13 = sig13[15651:]
    # sig27 = sig27[15651:]
    # sig14 = sig14[15651:]
    # sig28 = sig28[15651:]

    print('ok')