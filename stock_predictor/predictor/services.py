import numpy as np
import yfinance as yf
from os import path, makedirs
import matplotlib.pyplot as plt
from django.conf import settings
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


def get_stock_data(ticker, years=5):
    """Автоматически определяет дату начала торгов акции и загружает историю"""
    # Получаем информацию об акции
    stock_info = yf.Ticker(ticker)

    # Получаем историю с самой первой доступной даты
    hist = stock_info.history(period=f"{years}y")

    return hist['Close'].values.reshape(-1, 1), hist.index


def moving_average(prices, method='moving_average'):
    """Скользящее среднее"""
    if method == 'moving_average':
        # Скользящее среднее за последние 30 дней
        return float(np.mean(prices[-30:]))

    elif method == 'recent_trend':
        # Линейная регрессия только на последних 90 днях
        if len(prices) > 90:
            recent_prices = prices[-90:]
            days = np.arange(len(recent_prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(days, recent_prices)
            return float(model.predict([[len(recent_prices) + 30]])[0][0])
        else:
            return prices[-1][0]  # Последняя цена

    elif method == 'last_price':
        # Просто возвращаем последнюю цену
        return prices[-1][0]


def create_prediction_plot(ticker, historical_prices, historical_dates, future_price, method_used):
    """Создает график с реалистичным прогнозом"""
    plt.figure(figsize=(12, 6))

    # Исторические данные
    plt.plot(historical_dates, historical_prices.flatten(),
             label='Исторические данные', linewidth=2, color='blue')

    # Последняя цена
    last_date = historical_dates[-1]
    last_price = historical_prices[-1][0]

    # Показываем текущую цену
    plt.scatter(last_date, last_price, color='green', s=100, label=f'Текущая цена: ${last_price:.2f}')

    # Прогноз (пунктирной линией)
    future_date = last_date + timedelta(days=30)
    plt.scatter(future_date, future_price, color='red', s=100, label=f'Прогноз ({method_used}): ${future_price:.2f}')

    # Соединяем линией
    plt.plot([last_date, future_date], [last_price, future_price], 'r--', alpha=0.7)

    plt.title(f'Акция {ticker}: текущая ${last_price:.2f}, прогноз ${future_price:.2f}')
    plt.xlabel('Дата')
    plt.ylabel('Цена ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Сохраняем график
    plot_filename = f'{ticker}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plot_path = path.join(settings.MEDIA_ROOT, 'plots', plot_filename)
    makedirs(path.dirname(plot_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return f'/media/plots/{plot_filename}'

def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model
