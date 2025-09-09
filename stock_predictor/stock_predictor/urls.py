from django.contrib import admin
from django.conf import settings
from django.urls import path, include
from django.conf.urls.static import static


# Основные URL-маршруты проекта
# Определяет структуру всего веб-приложения

urlpatterns = [
    # Административная панель Django доступна по адресу: http://127.0.0.1:8000/admin/
    path('admin/', admin.site.urls),

    # Включение URL-маршрутов из приложения predictor
    # Все маршруты из predictor/urls.py теперь доступны от корня сайта
    path('', include('predictor.urls')),
]

# Добавление поддержки медиа-файлов в режиме разработки позволяет обслуживать загруженные файлы (графики) через Django
# В production следует использовать веб-сервер (Nginx/Apache) для медиа-файлов
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
