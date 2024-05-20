import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import matplotlib.pyplot as plt

# App title
st.title("Stock Market Price Prediction")

# Sidebar
st.sidebar.header("User Input")
selected_stock = st.sidebar.text_input("Enter stock ticker", "AAPL")
start_date = st.sidebar.date_input("Start date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())


# Fetch stock data
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error("No data found for the given stock ticker and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


data = fetch_data(selected_stock, start_date, end_date)

if data is not None:
    # Ensure we get the last 5 business days
    last_5_business_days = pd.bdate_range(end=end_date, periods=5).strftime('%Y-%m-%d').tolist()
    recent_data = data[data.index.isin(last_5_business_days)]

    # Display stock data
    st.subheader(f"Stock data for {selected_stock}")
    data.index = data.index.strftime('%Y-%m-%d')
    st.write(recent_data)  # Show the last 5 business days of stock data

    # Calculate moving averages
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    # Prepare data for prediction
    data['Date'] = pd.to_datetime(data.index)
    data['Date'] = data['Date'].map(datetime.datetime.toordinal)

    X = np.array(data['Date']).reshape(-1, 1)
    y = np.array(data['Close'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate error
    mse = mean_squared_error(y_test, y_pred)
    mse_rounded = round(mse, 2)
    st.write(f"Mean Squared Error: {mse_rounded}")

    # Plot the predictions and moving averages
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Prices')
    ax.plot(X_test, y_pred, color='red', label='Predicted Prices')
    ax.plot(data['Date'], data['SMA50'], color='green', label='50-day SMA')
    ax.plot(data['Date'], data['SMA200'], color='purple', label='200-day SMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.set_title(f"{selected_stock} Price Prediction")
    st.pyplot(fig)

    # Predict future prices
    future_dates = pd.date_range(start=end_date, periods=30)
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    future_pred = model.predict(future_dates_ordinal)
    future_pred_rounded = np.round(future_pred, 2)

    # Display future predictions
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_pred_rounded})
    future_data['Date'] = future_data['Date'].dt.strftime('%Y-%m-%d')
    future_data.set_index('Date', inplace=True)

    st.subheader("Future Price Predictions")
    st.dataframe(future_data)  # Display in a wider format using st.dataframe()

    # Add download button for stock data
    st.download_button(
        label="Download Stock Data as CSV",
        data=data.to_csv().encode('utf-8'),
        file_name=f'{selected_stock}_data.csv',
        mime='text/csv'
    )

    # Add download button for future predictions
    st.download_button(
        label="Download Future Predictions as CSV",
        data=future_data.to_csv().encode('utf-8'),
        file_name=f'{selected_stock}_future_predictions.csv',
        mime='text/csv'
    )
else:
    st.write("Please enter a valid stock ticker and date range.")
