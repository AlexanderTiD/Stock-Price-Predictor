import numpy as np
from django.shortcuts import render
from stock_predictor.backend.api import get_stock_data, train_model


def predict_view(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'AAPL')
        prices = get_stock_data(ticker, '2020-01-01', '2023-01-01')
        days = np.arange(len(prices)).reshape(-1, 1)
        model = train_model(days, prices)
        future_price = model.predict(np.array([[len(prices) + 30]]))[0][0]

        return render(request, 'predictor/result.html', {
            'ticker': ticker,
            'future_price': round(future_price, 2)
        })

    return render(request, 'predictor/form.html')
