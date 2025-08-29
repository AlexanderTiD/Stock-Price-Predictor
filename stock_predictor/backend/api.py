import yfinance as yf
from sklearn.linear_model import LinearRegression

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model
