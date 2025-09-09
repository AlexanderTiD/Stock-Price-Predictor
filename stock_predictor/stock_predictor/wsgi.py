import os
from django.core.wsgi import get_wsgi_application

# Настройка WSGI для проекта stock_predictor
# WSGI (Web Server Gateway Interface) используется для традиционных синхронных серверов типа Gunicorn

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor.settings')

application = get_wsgi_application()
