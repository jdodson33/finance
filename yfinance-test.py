import numpy as np
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def yfinance_data_capture(start, end, tickers, file_path):

    dataframes = []

    for ticker in tickers:
        df = yf.download(tickers=ticker, start=start, end=end, interval='1m')
        df.reset_index(inplace=True)
        df["Datetime"] = df["Datetime"].dt.strftime('%Y/%m/%d %H:%M:%S')
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        dataframes.append(df)

    with pd.ExcelWriter(file_path) as writer:

        for i, ticker in enumerate(tickers):
            dataframes[i].to_excel(writer, sheet_name=ticker, index=False)


# start = datetime(2023, 10, 25)
# end = datetime(2023, 10, 30)
# file_path = "C:\\finance\\yfinance data 2.xlsx"
# tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "UNH", "JNJ"]
# yfinance_data_capture(start, end, tickers, file_path)


def finance_data_analysis(file_path):
    from sklearn.linear_model import LinearRegression
    import numpy as np

    xlsx = pd.ExcelFile(file_path)
    df_dict = pd.read_excel(xlsx, None)

    AAPL_dataset = df_dict["AAPL"]

    date_data, price_data = list(AAPL_dataset["Datetime"]), list(AAPL_dataset["Close"])
    n = int(0.8 * len(date_data))

    train_dict = {
        'Date': date_data[:n],
        'Price': price_data[:n]
    }

    test_dict = {
        'Date': date_data[n:],
        'Price': price_data[n:]
    }

    train_data = pd.DataFrame(train_dict)
    test_data = pd.DataFrame(test_dict)

    train_x, train_y = np.array(range(len(train_data["Date"]))).reshape(-1, 1), np.array(train_data["Price"]).reshape(-1, 1)
    test_x, test_y = np.array(range(len(test_data['Price']))).reshape(-1, 1), np.array(test_data["Price"]).reshape(-1, 1)

    engine = LinearRegression()
    model = engine.fit(train_y, train_x)

    predicted_price = engine.predict(test_x)

    print(predicted_price, test_x)

    plt.plot(range(len(date_data)), price_data, label='Actual', color='g')
    plt.plot(range(len(test_x)), predicted_price, label='Predicted', color='b')

    plt.ylim(170, 180)
    # plt.xlim(datetime(2023, 10, 17), datetime(2023, 10, 28))

    plt.show()


#excel_file = "C:\\finance\\yfinance data.xlsx"
#finance_data_analysis(excel_file)


def fiance_machine_learning(file_path):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import linear_model

    from keras.layers import LSTM, Dense, Dropout
    from keras.models import Sequential
    import keras.backend as k
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam
    from keras.models import load_model

    df = pd.read_excel(file_path, sheet_name='AAPL', index_col='Datetime', parse_dates=True)

    output_var = pd.DataFrame(df['Adj Close'])
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)

    timesplit = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[
                                                                len(train_index): (len(train_index) + len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (
                    len(train_index) + len(test_index))].values.ravel()

    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')

    history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

    y_pred = lstm.predict(X_test)

    y_pred2 = lstm.predict(X_train)

    plt.subplot(211)
    plt.plot(y_test, label='TrueValue')
    plt.plot(y_pred, label='LSTMValue')
    plt.title('Prediction by LSTM')
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.legend()

    plt.subplot(212)
    plt.plot(y_pred2, label='LSTM Train Values')
    plt.plot(y_train, label='Actual')

    plt.show()


excel_file = "C:\\finance\\yfinance data.xlsx"
fiance_machine_learning(excel_file)