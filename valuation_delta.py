import yfinance as yf

msft_ticker = yf.Ticker("MSFT")
aapl_ticker = yf.Ticker("AAPL")

msft_price = msft_ticker.info.get('regularMarketPrice', msft_ticker.info.get('currentPrice'))
aapl_price = aapl_ticker.info.get('regularMarketPrice', aapl_ticker.info.get('currentPrice'))

msft_shares = msft_ticker.info.get('sharesOutstanding', 7.43e9) / 1e9  #
aapl_shares = aapl_ticker.info.get('sharesOutstanding', 15.04e9) / 1e9
msft_cap = round(msft_price * msft_shares, 2)
aapl_cap = round(aapl_price * aapl_shares, 2)

print(f"MSFT Shares: {msft_shares} billion")
print(f"MSFT Latest Price: {msft_price}")
print(f"MSFT mkt. cap: {msft_cap} billion")
print(f"AAPL Shares: {aapl_shares} billion")
print(f"AAPL Latest Price: {aapl_price}")
print(f"AAPL mkt. cap: {aapl_cap} billion")

msft_aapl_delta = round(msft_cap - aapl_cap, 3)
print(f"(MSFT - AAPL): {msft_aapl_delta} billon") #
