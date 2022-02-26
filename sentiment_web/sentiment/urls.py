from django.contrib import admin
from django.urls import path,include
from sentiment import views


urlpatterns = [
    path('', views.SentimentView.as_view()),
]
