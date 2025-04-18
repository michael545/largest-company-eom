import os
import yfinance as yf
import pandas as pd

def fetch_and_save_data(tickers, start_date, end_date, candle_length="1d", data_folder="data_folder"):

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for ticker in tickers:
        print(f"Fetching data for {ticker} with {candle_length} candles...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=candle_length, auto_adjust=True)

            if data.empty:
                print(f"No data fetched for {ticker}. Skipping...")
                continue

            # Ensure 'Datetime' is a column and clean up the DataFrame
            data.reset_index(inplace=True)
            if "Adj Close" in data.columns:
                data = data.rename(columns={"Date": "Datetime", "Adj Close": "Price"})
            else:
                data = data.rename(columns={"Date": "Datetime", "Close": "Price"})

            required_columns = ["Datetime", "Price", "High", "Low", "Open", "Volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Missing columns {missing_columns} for {ticker}. Skipping...")
                continue

            data = data[required_columns]

            # Drop rows with missing values
            data.dropna(inplace=True)

            file_path = os.path.join(data_folder, f"{ticker}.csv")
            data.to_csv(file_path, index=False)
            print(f"Saved {ticker} data to {file_path}")

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

if __name__ == "__main__":
    # config
    tickers = ["MSFT", "AAPL", "SPY"]
    start_date = "2025-01-01"
    end_date = "2025-04-04"
    candle_length = "1d" 

    fetch_and_save_data(tickers, start_date, end_date, candle_length, "data")