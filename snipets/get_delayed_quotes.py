import time
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum # TickTypeのenum値を使うためにインポート
from ibapi.utils import iswrapper

class DelayedQuoteFetcher(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_id_event = Event()
        self.error_event = Event()
        self.error_message = ""
        self.current_req_id = 0 # reqMktData に渡す tickerId を保持

        self.bid_price = None
        self.ask_price = None
        self.last_price = None
        self.bid_size = None
        self.ask_size = None
        self.quote_data_received = Event() # 何かしらの気配値データを受信したか

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        # nextValidIdは注文ID用だが、リクエストIDのベースとしても使える
        # self.current_req_id = orderId # ここでは reqMktData の tickerId は別途設定
        self.next_valid_id_event.set()

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # 接続情報や一般的な通知は無視
        # Code 10167 は「遅延データを表示します」という情報なので、エラーイベントをセットしない
        if errorCode in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101, 2119, 10167]: # 10167 を無視リストに追加
            if errorCode == 10167:
                # このメッセージは情報提供なので、エラーとして扱わない
                print(f"INFO Message (Not an error for delayed data): Id: {reqId}, Code: {errorCode}, Msg: {errorString}")
                # self.error_event.set() # ここではセットしない
                return
            # 321は購読エラーだが、遅延データでも出る可能性があるのでここでは無視しない方が良いかも
            if errorCode == 321 and "Delayed market data is not subscribed" in errorString:
                 self.error_message = f"DelayedQuote Error (Subscription): Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
                 print(self.error_message)
                 if reqId == self.current_req_id:
                    self.error_event.set()
                 return
            print(f"INFO/DEBUG Message: Id: {reqId}, Code: {errorCode}, Msg: {errorString}")
            return

        self.error_message = f"DelayedQuote Error: Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
        print(self.error_message)
        if reqId == self.current_req_id or reqId == -1: # reqIdが-1のエラーも考慮
            self.error_event.set()

    @iswrapper
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        super().tickPrice(reqId, tickType, price, attrib)
        if reqId != self.current_req_id: return

        changed = False
        # TickTypeEnumの直接比較ではなく、整数値で比較
        if tickType == 1: # BID_PRICE
            self.bid_price = price
            print(f"DelayedQuote Update: BID Price = {price}")
            changed = True
        elif tickType == 2: # ASK_PRICE
            self.ask_price = price
            print(f"DelayedQuote Update: ASK Price = {price}")
            changed = True
        elif tickType == 4: # LAST_PRICE
            self.last_price = price
            print(f"DelayedQuote Update: LAST Price = {price}")
            changed = True
        
        if changed:
            # quote_data_received イベントの扱いを修正
            # データを受信するたびにセットし、メインループでチェック後にクリアする方が良い場合もある
            # ここでは、bidとaskが揃うまでメインループで待つので、セットするだけでよい
            if not self.quote_data_received.is_set() and self.bid_price is not None and self.ask_price is not None:
                 self.quote_data_received.set()


    @iswrapper
    def tickSize(self, reqId: int, tickType: int, size: int):
        super().tickSize(reqId, tickType, size)
        if reqId != self.current_req_id: return

        changed = False
        # TickTypeEnumの直接比較ではなく、整数値で比較
        if tickType == 0: # BID_SIZE
            self.bid_size = size
            print(f"DelayedQuote Update: BID Size = {size}")
            changed = True
        elif tickType == 3: # ASK_SIZE
            self.ask_size = size
            print(f"DelayedQuote Update: ASK Size = {size}")
            changed = True
        # LAST_SIZE (5) や VOLUME (8) も必要なら追加
        
        if changed:
            # bidとaskが揃うまでメインループで待つので、セットするだけでよい
            if not self.quote_data_received.is_set() and self.bid_price is not None and self.ask_price is not None:
                 self.quote_data_received.set()

def get_delayed_n225_futures_quote(
    expiration_month: str, # YYYYMM 形式
    duration_sec=15, # データ待機時間
    host="127.0.0.1", port=7497, clientId=9 # 他のクライアントIDと重複しないように
    ):
    """
    N225先物の遅延気配値を取得する試み。
    """
    fetcher = DelayedQuoteFetcher()
    print(f"Connecting to {host}:{port} with clientId:{clientId} for delayed quotes...")
    fetcher.connect(host, port, clientId)

    if not fetcher.isConnected():
        print("Connection failed.")
        return None

    api_thread = Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    # nextValidIdは必須ではないが、接続確認の一環として待つ
    if not fetcher.next_valid_id_event.wait(timeout=10):
        print("Timeout waiting for nextValidId (connection may be unstable).")
        # 続行を試みることもできるが、ここでは失敗扱い
        fetcher.disconnect()
        api_thread.join()
        return None

    # --- 遅延データタイプのリクエスト ---
    # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
    print("Requesting Market Data Type: 3 (Delayed)")
    fetcher.reqMarketDataType(3) # 遅延データをリクエスト

    # --- コントラクト定義 ---
    contract = Contract()
    contract.symbol = "N225"
    contract.secType = "FUT"
    contract.exchange = "OSE.JPN"
    contract.currency = "JPY"
    contract.lastTradeDateOrContractMonth = expiration_month # YYYYMM形式

    # --- マーケットデータリクエスト ---
    # tickerIdは一意であれば何でも良いが、nextValidIdをベースにすると管理しやすい
    # ここでは固定のIDを使う (例: 1001) か、nextValidIdからインクリメントする
    fetcher.current_req_id = 1001 # このリクエストのためのID
    
    print(f"Requesting delayed market data for {contract.symbol} {expiration_month} (ReqId: {fetcher.current_req_id})...")
    # genericTickList: "" はデフォルトセット (Bid, Ask, Last, Sizesなど)
    # snapshot: False でストリーミング (遅延データでも更新があれば来る)
    fetcher.reqMktData(fetcher.current_req_id, contract, "", False, False, [])

    print(f"Waiting for delayed quote data for up to {duration_sec} seconds...")
    
    wait_start_time = time.time()
    data_found = False
    while time.time() - wait_start_time < duration_sec:
        if fetcher.error_event.is_set():
            # エラーメッセージは error コールバック内で出力されるので、ここではループを抜けるだけ
            print(f"Error flag set, stopping wait. Check logs for: {fetcher.error_message}")
            break
        # quote_data_received イベントがセットされるか、bidとaskが揃うのを待つ
        if fetcher.quote_data_received.is_set() or (fetcher.bid_price is not None and fetcher.ask_price is not None):
            print("Sufficient quote data received or event triggered.")
            data_found = True
            break
        time.sleep(0.2) # ポーリング間隔

    if not data_found and not fetcher.error_event.is_set():
        print("Timeout waiting for complete quote data. Displaying what was received.")

    print("Cancelling market data subscription...")
    fetcher.cancelMktData(fetcher.current_req_id)

    fetcher.disconnect()
    api_thread.join(timeout=5)
    print("Disconnected from IB for delayed quotes.")

    result = {
        "symbol": contract.symbol,
        "expiration": expiration_month,
        "last_price": fetcher.last_price,
        "bid_price": fetcher.bid_price,
        "bid_size": fetcher.bid_size,
        "ask_price": fetcher.ask_price,
        "ask_size": fetcher.ask_size,
        "error": fetcher.error_message if fetcher.error_event.is_set() else None
    }
    
    print("\n--- Delayed Quote Data ---")
    for key, value in result.items():
        print(f"  {key.replace('_', ' ').capitalize()}: {value}")
    print("--------------------------")
    
    if result["bid_price"] is None and result["ask_price"] is None and result["last_price"] is None and not result["error"]:
        print("\nNote: No quote data was received. This could be because:")
        print("- Delayed data for N225 Futures is not available on your paper account.")
        print("- The market is closed and no end-of-day delayed data is provided for this instrument.")
        print("- Incorrect contract details or other API communication issues.")
    elif result["error"]:
         print(f"\nAn error occurred during data retrieval: {result['error']}")

    return result

if __name__ == '__main__':
    print("Attempting to fetch N225 Futures delayed quote...")
    
    # N225先物の直近限月などを指定 (YYYYMM形式)
    # 実際の取引可能な限月に合わせてください
    target_expiration = "202509"
    PORT = 4001 # 本番用ポート
    
    # clientIdは他のAPIクライアントと重複しないように
    quote_result = get_delayed_n225_futures_quote(target_expiration, port=PORT, clientId=106)

    # if quote_result:
    #     # ここで取得したデータを使って何か処理
    #     pass