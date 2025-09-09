from django.shortcuts import render
from .services import create_prediction_plot, get_stock_data, moving_average


def predict_view(request):
    """
    Обработчик представления для прогнозирования цен акций.

    Обрабатывает GET и POST запросы:
    - GET: Отображает форму ввода тикера акции
    - POST: Обрабатывает введенный тикер, получает данные,
            вычисляет прогноз и отображает результаты

    Аргументы:
        request (HttpRequest): Объект HTTP запроса от Django.

    Возвращает:
        HttpResponse: Ответ с HTML шаблоном формы или результатов.

    Шаблоны:
        - predictor/form.html: Форма ввода тикера
        - predictor/result.html: Страница с результатами прогноза
        - predictor/error.html: Страница ошибки при проблемах с данными

    Пример маршрута:
        http://127.0.0.1:8000/
    """
    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'AAPL').upper()

        try:
            # Получаем данные
            prices, dates = get_stock_data(ticker)

            if len(prices) < 30:
                return render(request, 'predictor/error.html', {
                    'error': f'Недостаточно данных для акции {ticker}'})

            # Прогноз
            current_price = float(prices[-1][0])
            future_price = moving_average(prices, method='moving_average')

            # Создаем график
            plot_url = create_prediction_plot(ticker, prices, dates, future_price, 'moving_average')

            return render(request, 'predictor/result.html', {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'future_price': round(future_price, 2),
                'plot_url': plot_url,
                'change_percent': round(((future_price - current_price) / current_price * 100), 2),
                'historical_data_points': len(prices),
                'first_date': f'{dates[0].strftime("%Y-%m-%d")}',
                'last_date':  f'{dates[-1].strftime("%Y-%m-%d")}'})

        except Exception as e:
            return render(request, 'predictor/error.html', {'error': f'Ошибка: {str(e)}'})
    return render(request, 'predictor/form.html')
