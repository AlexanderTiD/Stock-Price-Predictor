from . import views
from django.urls import path

# Конфигурация URL-маршрутов для приложения predictor.
# Определяет связь между URL-адресами и представлениями

urlpatterns = [
    # Корневой маршрут приложения.
    # Обрабатывает главную страницу с формой ввода и результатами прогноза
    path('', views.predict_view, name='predict'),

    # Пример добавления дополнительных маршрутов:
    # path('history/', views.prediction_history, name='history'),
    # path('api/predict/', views.api_predict, name='api_predict'),
]
