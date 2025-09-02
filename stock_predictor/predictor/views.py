from django.shortcuts import render
from .services import create_prediction_plot, get_stock_data, moving_average


def predict_view(request):
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
                'historical_data_points': len(prices),
                'first_date': f'{dates[0].strftime("%Y-%m-%d")}',
                'last_date':  f'{dates[-1].strftime("%Y-%m-%d")}'})

        except Exception as e:
            return render(request, 'predictor/error.html', {'error': f'Ошибка: {str(e)}'})
    return render(request, 'predictor/form.html')
