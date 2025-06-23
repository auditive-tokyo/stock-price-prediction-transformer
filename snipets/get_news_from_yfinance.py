import yfinance as yf
from datetime import datetime

def fetch_and_print_news(ticker_symbol):
    """指定されたティッカーシンボルのニュースを取得して表示する関数"""
    print(f"\n\n--- Fetching news for: {ticker_symbol} ---")
    
    # ティッカーオブジェクトを作成
    ticker = yf.Ticker(ticker_symbol)

    # .news 属性でニュースを取得
    news_list = ticker.news

    if news_list:
        # 取得したニュースを一つずつ表示
        for news_item in news_list:
            # .get()を使い、キーが存在しない場合は 'N/A' を表示してエラーを回避
            title = news_item.get('title', 'N/A')
            publisher = news_item.get('publisher', 'N/A')
            link = news_item.get('link', 'N/A')

            print(f"\n  Title: {title}")
            print(f"  Publisher: {publisher}")
            
            # タイムスタンプも安全に取得して表示
            if 'providerPublishTime' in news_item:
                try:
                    timestamp = datetime.fromtimestamp(news_item['providerPublishTime'])
                    print(f"  Published: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                except Exception as e:
                    print(f"  Could not parse timestamp: {e}")

            print(f"  Link: {link}")
        print("\n-----------------------------------------")
    else:
        print(f"No news found for {ticker_symbol}.")


if __name__ == '__main__':
    # テストしたいティッカーシンボルのリスト
    tickers_to_test = [
        "AAPL",  # Apple Inc. (米国株 - ニュースが豊富)
        "TSLA",  # Tesla, Inc. (米国株 - ニュースが豊富)
        "^N225"  # 日経平均株価 (比較用)
    ]

    for ticker in tickers_to_test:
        fetch_and_print_news(ticker)
