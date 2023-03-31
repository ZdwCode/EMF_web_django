from django.db import models


class LiquidInfo(models.Model):
    """
    页面高度
    """
    li_height = models.IntegerField()
    date = models.DateField(default='2022-01-01')


class LifeTimeInfo(models.Model):
    """
    lifeTime：剩余寿命0-100
    """
    lifeTime = models.IntegerField()


class StateInfo(models.Model):
    """
    systemType:系统状态
    networkType:网络状态
    runType:运行状态
    0：正常状态  1：不正常状态
    """
    systemType = models.IntegerField(default=0)
    networkType = models.IntegerField(default=0)
    runType = models.IntegerField(default=0)


class ThickInfo(models.Model):
    """
    主铁沟厚度
    """
    thickness = models.IntegerField()
    date = models.DateTimeField(default='2022-01-01 00:00:00')

class ThickInfo2(models.Model):
    """
    主铁沟厚度
    """
    thickness = models.IntegerField()
    date = models.DateTimeField(default='2022-01-01 00:00:00')

class ThickInfo3(models.Model):
    """
    主铁沟厚度
    """
    thickness = models.IntegerField()
    date = models.DateTimeField(default='2022-01-01 00:00:00')
class ThickInfo4(models.Model):
    """
    主铁沟厚度
    """
    thickness = models.IntegerField()
    date = models.DateTimeField(default='2022-01-01 00:00:00')


class LastId(models.Model):
    lastid_1 = models.IntegerField()
    lastid_2 = models.IntegerField()
    lastid_3 = models.IntegerField()
    lastid_4 = models.IntegerField()


class LastId2(models.Model):
    lastid_1 = models.IntegerField()
    lastid_2 = models.IntegerField()
    lastid_3 = models.IntegerField()
    lastid_4 = models.IntegerField()

class WarmInfo(models.Model):
    """
    预警设置
    """
    thickness_warm = models.IntegerField()
    height_warm = models.IntegerField()


class UserInfo(models.Model):
    """
    用户信息管理
    """
    name = models.CharField(max_length=32)
    username = models.CharField(max_length=32)
    password = models.CharField(max_length=32)


class DateTest(models.Model):
    date = models.DateField()


class DatePath(models.Model):
    path = models.CharField(max_length=128)