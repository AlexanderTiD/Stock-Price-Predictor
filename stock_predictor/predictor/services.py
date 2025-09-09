import numpy as np
import yfinance as yf
from os import path, makedirs
import matplotlib.pyplot as plt
from django.conf import settings
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


def get_stock_data(ticker, years=5):
    """
    Получает исторические данные по акциям для заданного тикера.

    Использует библиотеку yfinance для получения цен закрытия акции
    за указанное количество лет. Данные возвращаются в формате,
    подходящем для алгоритмов машинного обучения.

    Аргументы:
        ticker (str): Тикер акции (например, 'AAPL', 'TSLA', 'NVDA').
        years (int, опционально): Количество лет исторических данных. По умолчанию 5.

    Возвращает:
        tuple: Кортеж, содержащий два элемента:
            - prices (numpy.ndarray): Массив цен закрытия в формате для ML (-1, 1)
            - dates (pandas.DatetimeIndex): Соответствующие даты цен

    Пример:
        >>> prices, dates = get_stock_data('AAPL', years=3)
        >>> print(f"Получено {len(prices)} точек данных для AAPL")
    """
    # Получаем информацию об акции
    stock_info = yf.Ticker(ticker)

    # Получаем историю с самой первой доступной даты
    hist = stock_info.history(period=f"{years}y")

    return hist['Close'].values.reshape(-1, 1), hist.index

def moving_average(prices, method='moving_average'):
    """
    Прогнозирует будущую цену акции на основе исторических данных.

    Предоставляет несколько методов прогнозирования: скользящее среднее,
    линейная регрессия данных или просто последняя известная цена.

    Аргументы:
        prices (numpy.ndarray): Массив исторических цен акции.
        method (str, опционально): Метод прогнозирования:
            - 'moving_average': Скользящее среднее за 30 дней
            - 'recent_trend': Линейная регрессия на 90 днях
            - 'last_price': Последняя известная цена
            По умолчанию 'moving_average'.

    Возвращает:
        float: Прогнозируемая цена акции.

    Пример:
        >>> future_price = moving_average(prices, method='moving_average')
        >>> print(f"Прогнозируемая цена: ${future_price:.2f}")
    """
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
    """
    Создает визуализацию исторических данных и прогноза цены акции.

    Генерирует график с историческими ценами, текущей ценой и прогнозируемой
    ценой на 30 дней вперед. График сохраняется в медиа-директорию проекта.

    Аргументы:
        ticker (str): Тикер акции для заголовка графика.
        historical_prices (numpy.ndarray): Исторические цены акции.
        historical_dates (pandas.DatetimeIndex): Даты соответствующих цен.
        future_price (float): Прогнозируемая цена на 30 дней вперед.
        method_used (str): Использованный метод прогнозирования для легенды.

    Возвращает:
        str: URL путь к сохраненному изображению графика.

    Пример:
        >>> plot_url = create_prediction_plot('AAPL', prices, dates, 150.50, 'moving_average')
        >>> print(f"График сохранен: {plot_url}")
    """
    # Настройка стиля
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Основной график цен
    ax.plot(historical_dates, historical_prices.flatten(),
            color='#2962FF', linewidth=2, label='Исторические данные')

    # Текущая цена
    last_date = historical_dates[-1]
    last_price = historical_prices[-1][0]

    # Прогнозная точка
    future_date = last_date + timedelta(days=30)

    # Добавляем свечной стиль для последнего дня
    ax.scatter(last_date, last_price, color='#00E676', s=120, edgecolors='white', linewidth=2, zorder=5,
               label=f'Текущая: ${last_price:.2f}')

    # Прогнозная точка
    ax.scatter(future_date, future_price, color='#FF6D00', s=120, edgecolors='white', linewidth=2, zorder=5,
               label=f'Прогноз: ${future_price:.2f}')

    # Линия прогноза
    ax.plot([last_date, future_date], [last_price, future_price],
            color='#FF6D00', linestyle='--', linewidth=2, alpha=0.8)

    # Заполнение под графиком
    ax.fill_between(historical_dates, historical_prices.flatten(), alpha=0.2, color='#2962FF')

    # Настройка осей и сетки
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Цвета осей
    ax.spines['bottom'].set_color('#757575')
    ax.spines['top'].set_color('#757575')
    ax.spines['right'].set_color('#757575')
    ax.spines['left'].set_color('#757575')

    # Форматирование цен на оси Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))

    # Поворот дат для лучшей читаемости
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Заголовок и подписи
    ax.set_title(f'{ticker} - Текущая: ${last_price:.2f} | Прогноз: ${future_price:.2f}',
                 fontsize=16, fontweight='bold', pad=20, color='white')

    ax.set_xlabel('Дата', fontsize=12, color='#BDBDBD', labelpad=10)
    ax.set_ylabel('Цена ($)', fontsize=12, color='#BDBDBD', labelpad=10)

    # Расчет изменения цены
    change_percent = ((future_price - last_price) / last_price * 100)
    change_color = '#00E676' if change_percent >= 0 else '#FF5252'

    # Единая информационная панель с легендой и статистикой
    info_text = f'''📊 {ticker} - АНАЛИЗ
    💰 ЦЕНЫ:
    ┣ Текущая: ${last_price:.2f}
    ┣ Прогноз: ${future_price:.2f}
    ┗ Изменение: <span style="color:{change_color}">{change_percent:+.2f}%</span>

    📈 ДАННЫЕ:
    ┣ Период: {len(historical_prices)} дней
    ┣ Начало: {historical_dates[0].strftime("%d.%m.%Y")}
    ┗ Конец: {historical_dates[-1].strftime("%d.%m.%Y")}

    🔧 МЕТОД:
    ┗ {method_used}

    📋 ЛЕГЕНДА:
    ┣ Синяя линия - Исторические данные
    ┣ Зеленая точка - Текущая цена
    ┗ Оранжевая точка - Прогноз на 30 дней'''

    # Автоматическое позиционирование - находим свободное место
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    chart_width = x_range[1] - x_range[0]
    chart_height = y_range[1] - y_range[0]

    # Позиционируем в левом верхнем углу графика (не экрана)
    info_x = x_range[0] + chart_width * 0.02  # 2% от левого края
    info_y = y_range[1] - chart_height * 0.05  # 5% от верхнего края

    # Информационная панель
    ax.text(info_x, info_y, info_text, bbox=dict(boxstyle='round', facecolor='#424242',
                      alpha=0.9, edgecolor='#757575', pad=1), fontfamily='monospace', color='white', fontsize=10,
            verticalalignment='top', linespacing=1.4)

    # Настройка layout
    plt.tight_layout()

    # Сохранение с высоким качеством
    plot_filename = f'{ticker}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plot_path = path.join(settings.MEDIA_ROOT, 'plots', plot_filename)
    makedirs(path.dirname(plot_path), exist_ok=True)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#121212', edgecolor='none')
    plt.close()

    return f'/media/plots/{plot_filename}'

def train_model(x, y):
    """
    Обучает модель линейной регрессии на предоставленных данных.

    Создает и обучает модель линейной регрессии scikit-learn
    для прогнозирования цен на акции.

    Аргументы:
        x (numpy.ndarray): Признаки для обучения (обычно временные периоды).
        y (numpy.ndarray): Целевые значения (цены акций).

    Возвращает:
        LinearRegression: Обученная модель линейной регрессии.

    Пример:
        >>> model = train_model(days, prices)
        >>> prediction = model.predict([[100]])
    """
    model = LinearRegression()
    model.fit(x, y)
    return model
