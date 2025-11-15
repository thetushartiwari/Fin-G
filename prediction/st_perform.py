import yfinance as yf


SECTOR_MAPPING = {### diversifying the stock in different categories
    "tech": {
        "low_risk": ["MSFT", "IBM", "ORCL"],
        "moderate_risk": ["AAPL", "GOOGL", "ADBE"],
        "high_risk": ["NVDA", "TSLA", "META"]
    },
    "finance": {
        "low_risk": ["JPM", "WFC", "BAC"],
        "moderate_risk": ["GS", "MS", "AXP"],
        "high_risk": ["SQ", "PYPL", "COIN"]
    },
    "healthcare": {
        "low_risk": ["JNJ", "PFE", "MRK"],
        "moderate_risk": ["UNH", "ABBV", "AMGN"],
        "high_risk": ["MRNA", "BIIB", "VRTX"]
    },
    "fmcg": {
        "low_risk": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"],
        "moderate_risk": ["BRITANNIA.NS", "DABUR.NS", "MARICO.NS"],
        "high_risk": ["UBL.NS", "TATACONSUM.NS", "GODREJCP.NS"]
    }
}

def get_top_stocks(interest, risk_score):
    category = "low_risk" if risk_score <= 3 else "moderate_risk" if risk_score <= 6 else "high_risk"
    stocks = SECTOR_MAPPING.get(interest.lower(), {}).get(category, [])
    stock_data = []

    for symbol in stocks:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")

        if hist.empty:
            continue  

        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        stock_return = ((end_price - start_price) / start_price) * 100  # Calculate 6-month return

        stock_data.append((symbol, float(round(stock_return, 2))))  

    # Sort stocks by return percentage in descending order and return top recommendations
    sorted_stocks = sorted(stock_data, key=lambda x: x[1], reverse=True)
    return sorted_stocks[:3]  # Top 3 recommendations per risk category

##if __name__ == "__main__":
    ##interest = 'tech'
    ##risk_score = 8
    ##print(get_top_stocks(interest,risk_score))
