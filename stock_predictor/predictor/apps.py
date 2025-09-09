from django.apps import AppConfig


class PredictorConfig(AppConfig):
    """
    Конфигурационный класс для приложения predictor.

    Определяет основные настройки приложения, включая тип поля
    для автоматического создания первичных ключей и имя приложения.

    Атрибуты:
        default_auto_field (str): Тип поля для автоматических первичных ключей.
        name (str): Полный Python путь к приложению.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'
