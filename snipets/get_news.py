import time
from threading import Thread, Event
from datetime import datetime, timedelta

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract # ContractDetailsをリクエストする場合や、銘柄指定に使う
# from ibapi.news import NewsProvider, NewsTick # HistoricalNewsでは直接使わないかも
from ibapi.utils import iswrapper

class HistoricalNewsFetcher(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_id_event = Event()
        self.historical_news_event = Event()
        self.historical_news_end_event = Event()
        self.error_event = Event()
        self.error_message = ""
        self.current_req_id = 0
        self.news_articles = []

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_req_id = orderId
        self.next_valid_id_event.set()

    @iswrapper
    def historicalNews(self, reqId: int, time: str, providerCode: str, articleId: str, headline: str):
        super().historicalNews(reqId, time, providerCode, articleId, headline)
        news_item = {
            "reqId": reqId,
            "time": time,
            "providerCode": providerCode,
            "articleId": articleId,
            "headline": headline
        }
        print(f"Historical News: Time={time}, Provider={providerCode}, ID={articleId}, Headline={headline[:100]}...")
        self.news_articles.append(news_item)
        self.historical_news_event.set() # 何か受信したことを通知

    @iswrapper
    def historicalNewsEnd(self, reqId: int, hasMore: bool):
        super().historicalNewsEnd(reqId, hasMore)
        print(f"HistoricalNewsEnd. ReqId: {reqId}, HasMore: {hasMore}")
        self.historical_news_end_event.set()

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2105, 2106, 2107, 2108, 2158]: # 接続メッセージは無視
            self.error_message = f"Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
            print(self.error_message)
            if reqId == self.current_req_id : # 自分のリクエストに対するエラー
                 self.error_event.set()


def get_yesterdays_n225_news(
    host="127.0.0.1", port=7497, clientId=4,
    n225_conId=0, # N225のconId (TWSで調べるか、ContractDetailsで取得)
    provider_codes="ALL" # 例: "REUTERS,DJ", または "ALL" (購読範囲内)
    ):
    """
    昨日のN225関連の過去ニュースヘッドラインを取得する試み。
    動作には適切なconId、provider_codes、そしてニュース購読が必要。
    """
    fetcher = HistoricalNewsFetcher()
    print(f"Connecting to {host}:{port} with clientId:{clientId} for historical news...")
    fetcher.connect(host, port, clientId)

    if not fetcher.isConnected():
        print("Connection failed.")
        return []

    api_thread = Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    if not fetcher.next_valid_id_event.wait(timeout=10):
        print("Timeout waiting for nextValidId.")
        fetcher.disconnect()
        api_thread.join()
        return []

    if n225_conId == 0:
        print("Error: n225_conId is not set. Cannot request historical news without a contract ID.")
        fetcher.disconnect()
        api_thread.join()
        return []

    # 昨日のおおよその日付範囲を設定
    # IBKR APIの日付形式 'yyyyMMdd HH:mm:ss z' (zはタイムゾーン、省略可)
    # タイムゾーンはTWSの設定に依存することがあるので注意
    yesterday = datetime.now() - timedelta(days=1)
    start_time_str = yesterday.strftime("%Y%m%d") + " 00:00:00"
    end_time_str = yesterday.strftime("%Y%m%d") + " 23:59:59"
    # または、より広い範囲や特定のタイムゾーンを指定
    # start_time_str = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d %H:%M:%S")
    # end_time_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d %H:%M:%S")


    print(f"Requesting historical news for conId {n225_conId} from {start_time_str} to {end_time_str} for providers '{provider_codes}' with reqId {fetcher.current_req_id}...")
    
    fetcher.reqHistoricalNews(
        reqId=fetcher.current_req_id,
        conId=n225_conId,
        providerCodes=provider_codes, # カンマ区切りまたは "ALL"
        startDateTime=start_time_str,
        endDateTime=end_time_str,
        totalResults=100, # 最大取得件数
        historicalNewsOptions=[] # 通常は空
    )

    # historicalNewsEnd またはエラーの受信を待つ
    while not (fetcher.historical_news_end_event.is_set() or fetcher.error_event.is_set()):
        time.sleep(0.1)
        if not api_thread.is_alive():
            print("API thread terminated unexpectedly.")
            break
        # タイムアウト処理もここに追加可能

    articles = []
    if fetcher.historical_news_end_event.is_set() and not fetcher.error_event.is_set():
        articles = fetcher.news_articles
        print(f"Successfully fetched {len(articles)} historical news headlines.")
    elif fetcher.error_event.is_set():
        print(f"Failed to fetch historical news due to an error: {fetcher.error_message}")
    else:
        print("Failed to fetch historical news (timeout or unknown issue).")

    fetcher.disconnect()
    api_thread.join(timeout=5)
    print("Disconnected from IB for historical news.")
    return articles

class RealtimeNewsStreamer(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_id_event = Event()
        self.error_event = Event()
        self.error_message = ""
        self.current_req_id = 0
        self.received_headlines = []
        self.news_tick_event = Event()

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_req_id = orderId
        self.next_valid_id_event.set()

    @iswrapper
    def tickNews(self, tickerId: int, timeStamp: int, providerCode: str, articleId: str, headline: str, extraData: str):
        super().tickNews(tickerId, timeStamp, providerCode, articleId, headline, extraData)
        news_item = {
            "tickerId": tickerId,
            "timeStamp": timeStamp, # Unix timestamp
            "providerCode": providerCode,
            "articleId": articleId,
            "headline": headline,
            "extraData": extraData
        }
        # timeStampを人間が読める形式に変換 (オプション)
        dt_object = datetime.fromtimestamp(timeStamp/1000) # IBKRはミリ秒単位のことがある
        formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Realtime News Tick: Time={formatted_time}, Provider={providerCode}, ID={articleId}, Headline={headline[:100]}...")
        self.received_headlines.append(news_item)
        self.news_tick_event.set() # 何か受信したことを通知

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101]: # 接続関連メッセージは無視
            self.error_message = f"Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
            print(self.error_message)
            if reqId == self.current_req_id:
                self.error_event.set()

def stream_realtime_news_headlines(
    news_provider_symbol: str, # 例: "BRFG:BRFG_ALL" (購読が必要)
    news_provider_exchange: str, # 例: "BRFG"
    duration_sec=30,
    host="127.0.0.1", port=7497, clientId=5
    ):
    """
    指定されたニュースプロバイダーからリアルタイムニュースヘッドラインをストリーミングする試み。
    動作にはニュースプロバイダーの購読が必要。
    """
    streamer = RealtimeNewsStreamer()
    print(f"Connecting to {host}:{port} with clientId:{clientId} for realtime news streaming...")
    streamer.connect(host, port, clientId)

    if not streamer.isConnected():
        print("Connection failed.")
        return []

    api_thread = Thread(target=streamer.run, daemon=True)
    api_thread.start()

    if not streamer.next_valid_id_event.wait(timeout=10):
        print("Timeout waiting for nextValidId.")
        streamer.disconnect()
        api_thread.join()
        return []

    contract = Contract()
    contract.symbol = news_provider_symbol
    contract.secType = "NEWS"
    contract.exchange = news_provider_exchange
    
    # リクエストIDはユニークである必要がある
    # streamer.current_req_id を使うか、新しいIDを割り当てる
    mkt_data_req_id = streamer.current_req_id + 1 # 例

    print(f"Requesting realtime news stream for {contract.symbol} on {contract.exchange} (reqId: {mkt_data_req_id})...")
    # Generic Tick Type "292" はニュースティック用
    # "mdoff,292" は、他のマーケットデータをオフにし、ニュースティックのみを要求する意図
    # ただし、"mdoff" が全てのケースで有効かはプロバイダーによる可能性あり
    # 単純に "292" だけを指定することも可能
    streamer.reqMktData(mkt_data_req_id, contract, "292", False, False, [])

    print(f"Streaming news for {duration_sec} seconds...")
    time.sleep(duration_sec)

    print("Cancelling news stream...")
    streamer.cancelMktData(mkt_data_req_id)

    streamer.disconnect()
    api_thread.join(timeout=5)
    print("Disconnected from IB for realtime news streaming.")

    if streamer.received_headlines:
        print(f"Streamed {len(streamer.received_headlines)} news headlines.")
    else:
        print("No news headlines streamed. Check subscription and provider details.")
        if streamer.error_message:
             print(f"Last error: {streamer.error_message}")
             
    return streamer.received_headlines

if __name__ == '__main__':
    # --- Historical News のテスト (既存のコード) ---
    print("Fetching yesterday's N225 historical news headlines (conceptual)...")
    target_conId = 531965322
    news_providers_for_historical = "RTRS,DJNWS" # 購読しているプロバイダーコードに置き換える

    if target_conId == 0:
        print("\nPlease set a valid 'target_conId' for N225 Index in the script for historical news.")
    elif not news_providers_for_historical or news_providers_for_historical == "ALL":
        print("\nPlease specify valid, subscribed news provider codes for historical news.")
    else:
        print(f"Using conId: {target_conId} for NIKKEI 225 INDEX (OSE.JPN) for historical news.")
        print(f"Requesting historical news from providers: {news_providers_for_historical}")
        historical_news_list = get_yesterdays_n225_news(
            port=7497,
            clientId=101,
            n225_conId=target_conId,
            provider_codes=news_providers_for_historical
        )
        if historical_news_list:
            print("\n--- Fetched Historical News Headlines ---")
            for i, article in enumerate(historical_news_list):
                print(f"\nArticle #{i+1}:")
                print(f"  Time: {article['time']}")
                print(f"  Provider: {article['providerCode']}")
                print(f"  Article ID: {article['articleId']}")
                print(f"  Headline: {article['headline']}")
            print("------------------------------------")
        else:
            print("\nNo historical news headlines were fetched.")

    print("\n" + "="*50 + "\n")

    # --- Realtime News Streaming のテスト ---
    print("Attempting to stream realtime news headlines (conceptual)...")
    # !!! 以下のシンボルと取引所は Briefing.com の例であり、購読が必要です !!!
    # !!! 無料で試せるニュースソースがあれば、それに置き換えてください !!!
    rt_news_symbol = "BRFG:BRFG_ALL" # 例: Briefing.com (有料)
    rt_news_exchange = "BRFG"       # 例: Briefing.com (有料)
    # rt_news_symbol = "NYSE:NEWS_ALL" # ダミー: もしNYSEの無料ニュースがあれば
    # rt_news_exchange = "NYSE"        # ダミー

    print(f"Note: Realtime news streaming for '{rt_news_symbol}' on '{rt_news_exchange}' likely requires a paid subscription.")
    
    # clientId は他と重複しないように
    streamed_headlines = stream_realtime_news_headlines(
        news_provider_symbol=rt_news_symbol,
        news_provider_exchange=rt_news_exchange,
        duration_sec=20, # 20秒間ストリーミングを試す
        clientId=102
    )

    if streamed_headlines:
        print("\n--- Streamed Realtime News Headlines ---")
        for i, news in enumerate(streamed_headlines):
            dt_object = datetime.fromtimestamp(news['timeStamp']/1000)
            formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nHeadline #{i+1} at {formatted_time}:")
            print(f"  Provider: {news['providerCode']}")
            print(f"  Article ID: {news['articleId']}")
            print(f"  Headline: {news['headline']}")
        print("---------------------------------------")
    else:
        print("\nNo realtime news headlines were streamed.")
