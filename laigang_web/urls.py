"""laigang_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from demo01 import views
from laigang_web import settings

urlpatterns = [
    #path('admin/', admin.site.urls),
    #path('index/', views.index),
    path('login/', views.login),
    #path('default/', views.default),
    path('info/ditch/', views.ditchInfo),
    path('info/liquid/', views.liquidInfo),
    path('info/iron/', views.ironInfo),
    path('info/warn/', views.warn),
    path('info/history/', views.historyInfo),
    path('info/user/', views.userInfo),
    path('datatest/', views.datatest),
    path('getLife/', views.getLifeDate),
    path('getState/', views.getStateData),
    path('getThick/', views.getThickData1),
    path('getThick2/', views.getThickData2),
    path('getThick3/', views.getThickData3),
    path('getThick4/', views.getThickData4),
    path('getHeight/', views.getLiquidData),
    path('warmEdit/', views.warmEdit),
    path('mytest/', views.test),
    path('changedate/', views.changedate)

]

# from django.conf.urls import static
# urlpatterns += static.static(settings.STATIC_URL, document_root=settings.STATIC)

