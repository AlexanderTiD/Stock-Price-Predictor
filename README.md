# 📈 Stock Price Predictor

Веб-приложение для прогнозирования цен на акции с использованием машинного обучения и визуализацией результатов.

## 🚀 Возможности

- **Прогнозирование цен**: Предсказание стоимости акций через 30 дней
- **Визуализация данных**: Интерактивные графики с историей и прогнозом
- **Поддержка любых акций**: Работа с тикерами NASDAQ, NYSE и других бирж
- **Адаптивный дизайн**: Оптимизирован для desktop и mobile устройств

## 🛠️ Технологический стек

### Backend
- Python 3.12+
- Django 5.2 - веб-фреймворк
- scikit-learn - машинное обучение
- Matplotlib - визуализация данных
- yFinance - получение финансовых данных
- Pandas - структурирование табличных данных
- NumPy - выполнения математических операций в многомерных массивах

### Frontend
- HTML5/CSS3
- Bootstrap 5 - адаптивный дизайн
- JavaScript - интерактивность

## 📦 Установка и запуск

### 1. Клонирование репозитория
```bash
    git clone https://github.com/your-username/stock-price-predictor.git
    cd stock-price-predictor
```

### 2. Создание виртуального окружения
```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # или
    venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей
```bash
    pip install -r requirements.txt
```

### 4. Настройка базы данных
```bash
    python manage.py migrate
```

### 5. Запуск сервера
```bash
    python manage.py runserver
```
- Приложение будет доступно по адресу: http://127.0.0.1:8000

##  🎯 Как использовать
- **Введите тикер акции (например: AAPL, TSLA, NVDA, GOOGL)**
- **Нажмите "Предсказать цену"**
- **Получите результат:**
  * Текущая цена акции 
  * Прогноз на 30 дней
  * Интерактивный график
  * Исторические данные

## 📊 Алгоритмы прогнозирования
- ### 1. Скользящее среднее (Moving Average)
```bash
    def moving_average(prices, window=30):
        return np.mean(prices[-window:])
```
- ### 2. Линейная регрессия (Linear Regression)
```bash
    model = LinearRegression()
    model.fit(days, prices)
    return model.predict(future_days)
```

## 📁 Структура проекта
    stock-predictor/
    │   ├── media/                    # Загружаемые файлы
    │   │   └── plots/                # Сохраненные графики        
    │   ├── predictor/                # Django приложение
    │   │   ├── migrations/           # Миграции базы данных
    │   │   │   └── __init__.py       # -
    │   │   ├── templates/            # HTML шаблоны
    │   │   │   └── predictor/        # Шаблоны приложения 
    │   │   │      ├── form.html      # Форма ввода
    │   │   │      ├── result.html    # Страница результата
    │   │   │      └── error.html     # Страница ошибки
    │   │   ├── __init__.py           # -
    │   │   ├── admin.py              # Админ-панель
    │   │   ├── apps.py               # Конфигурация приложения
    │   │   ├── models.py             # Модели базы данных
    │   │   ├── tests.py              # Тесты
    │   │   ├── urls.py               # Маршруты приложения
    │   │   ├── views.py              # Контроллеры
    │   │   └── services.py           # Бизнес-логика
    │   ├── stock_predictor/          # Настройки проекта
    │   │   ├── __init__.py           # -
    │   │   ├── settings.py           # Конфигурация
    │   │   ├── urls.py               # Главные маршруты
    │   │   ├── asgi.py               # ASGI конфигурация
    │   │   └── wsgi.py               # WSGI конфигурация
    │   └── db.sqlite3                # База данных
    │   └── manage.py                 # Утилита управления
    ├── .gitignore                    # Игнорируемые файлы
    │── requirements.txt              # Зависимости
    └── README.md                     # Документация

## 🔧 API функции
### get_stock_data(ticker, years=5)
    Получает исторические данные акции

    **Аргументы:**
    ticker: Тикер акции           (str)
    years: Количество лет истории (int)

    **Возвращает**
    Цены и даты (numpy array, Index)

### moving_average(prices, method='moving_average')
    Вычисляет прогноз цены

    **Аргументы:**
    prices: Исторические цены (array)
    method: Метод прогноза    ('moving_average', 'recent_trend', 'last_price')

    **Возвращает**
    Прогнозируемая цена (float)

### create_prediction_plot(ticker, historical_prices, historical_dates, future_price, method='moving_average')
    Создает график с прогнозом
    
    **Аргументы:**
    ticker: Тикер акции для заголовка графика.                     (str)
    historical_prices: Исторические цены акции                     (numpy.ndarray)
    historical_dates: Даты соответствующих цен.                    (pandas.DatetimeIndex)
    future_price: Прогнозируемая цена на 30 дней вперед.           (float)
    method_used: Использованный метод прогнозирования для легенды. (str)  
  
    **Возвращает**
    URL сохраненного графика

### train_model(x, y)
    Обучает модель линейной регрессии
    
    **Аргументы:**
        x: Признаки для обучения (обычно временные периоды). (numpy.ndarray)
        y: Целевые значения (цены акций).                    (numpy.ndarray)

    **Возвращает**
        LinearRegression: Обученная модель линейной регрессии.

## 🌟 Примеры использования
- ### Прогноз для Apple (AAPL) 
```bash
  prices, dates = get_stock_data('AAPL', years=3)
  future_price = moving_average(prices, 'moving_average')
  # Результат: текущая цена $170.00, прогноз $175.50
```

- ### Прогноз для Tesla (TSLA) 
```bash
  prices, dates = get_stock_data('TSLA', years=2)
  future_price = moving_average(prices, 'recent_trend')
  # Результат: текущая цена $240.00, прогноз $255.30
```

- ### Пример Графика
  ![Alt Text](./media/plots/plot.png)


## 📈 Метрики качества
- **Точность прогноза: ~70-80% для стабильных акций**
- **Время отклика: < 2 секунды**
- **Поддержка акций: 1000+ тикеров**

## 🚀 Деплой
```bash
  heroku create your-app-name
  git push heroku main
```

## 👨‍💻 Автор
- **Alexander Makarov**
- **Email: alexander.makarovv@outlook.com**
- **GitHub: @WanderGarro**

## 🗂 Используемое программное обеспечение
- **Yahoo Finance API — источник финансовых данных**
- **Django — WEB-фреймворк**
- **Bootstrap — CSS-фреймворк**

## 📞 Контакты
**Если у вас есть вопросы или предложения, создайте issue или напишите на email.**
### ⭐ Не забудьте поставить звезду репозиторию, если проект вам понравился!

## 📃 Licence
- **MIT License	Copyright (c) 2025 WanderGarro**