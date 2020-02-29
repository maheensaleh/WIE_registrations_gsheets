from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.base, name='base'),
    path('filter',views.filter,name = 'filter'),
    path('data_view/',views.data_view,name='data_view')

]