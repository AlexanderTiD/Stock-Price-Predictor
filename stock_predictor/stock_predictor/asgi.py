import os
from django.core.asgi import get_asgi_application

# Настройка ASGI для проекта stock_predictor
# ASGI (Asynchronous Server Gateway Interface) используется для асинхронных серверов типа Daphne или Uvicorn

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor.settings')

application = get_asgi_application()
