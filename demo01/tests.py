

import datetime
import random
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(date)
print(type(date))
someday = datetime.datetime(2023,6,10,22,33,32,7)
total = 730

for i in range(550):
    someday = someday + datetime.timedelta(hours=1)
    total = total-0.6 + random.randint(-5,5)
    result_time = someday.strftime("%Y-%m-%d %H:%M:%S")
    print(result_time,total)
    print(type(result_time))
# ThickInfo.objects.create(thickness=result, date=date)
# index1 = LastId.objects.get(id=1).lastid_1  # 201
# index1_new = index1 + 1
# LastId.objects.filter(id=1).update(lastid_1=index1_new)
# a_rand = random.randint(-6,6)
# b_rand = random.randint(-6,6)
# a = result+a_rand
# b = result+b_rand
# ThickInfo2.objects.create(thickness=a, date=date)
# index2 = LastId.objects.get(id=1).lastid_2  # 201
# index2_new = index2 + 1
# LastId.objects.filter(id=1).update(lastid_2=index2_new)
# ThickInfo3.objects.create(thickness=b, date=date)
# index3 = LastId.objects.get(id=1).lastid_3  # 201
# index3_new = index3 + 1
# LastId.objects.filter(id=1).update(lastid_3=index3_new)
# d = (result+a+b)/3
# ThickInfo4.objects.create(thickness=d, date=date)
# index4 = LastId.objects.get(id=1).lastid_4  # 201
# index4_new = index4 + 1
# LastId.objects.filter(id=1).update(lastid_4=index4_new)
# LastId2.objects.filter(id=1).update(lastid_1=last_id + 1)
# life_state = ((result-550)/950)*100
# LifeTimeInfo.objects.create(lifeTime=life_state)