from . import views
from django.urls import path


urlpatterns = [
    path('', views.predict_view, name='predict'),
]
