import time
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class MarketDepthFetcher(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_id_event = Event()
        self.error_event = Event()
        self.error_message = ""
        self.current_req_id = 0
        # 板情報を格納するためのデータ構造 (例:辞書)
        self.market_depth_bids = {} # {price: size}
        self.market_depth_asks = {} # {price: size}
        self.mkt_depth_data_event = Event() # データ受信の確認用

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_req_id = orderId
        self.next_valid_id_event.set()

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101]:
            self.error_message = f"MktDepth Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
            print(self.error_message)
            if reqId == self.current_req_id or reqId == self.current_req_id +1 : # 自分のリクエストIDに関連するエラー
                self.error_event.set()

    @iswrapper
    def updateMktDepth(self, reqId: int, position: int, operation: int, side: int, price: float, size: int):
        super().updateMktDepth(reqId, position, operation, side, price, size)
        
        side_str = "ASK" if side == 0 else "BID"
        op_str = ""
        if operation == 0: # Insert
            op_str = "INSERT"
        elif operation == 1: # Update
            op_str = "UPDATE"
        elif operation == 2: # Delete
            op_str = "DELETE"
        
        # 簡単な表示例 (実際にはもっと洗練されたデータ構造と表示が必要)
        print(f"MktDepth Update: ReqId={reqId}, Pos={position}, Op={op_str}({operation}), Side={side_str}({side}), Price={price}, Size={size}")

        # ここで self.market_depth_bids や self.market_depth_asks を更新するロジックを実装
        # 例:
        # target_book = self.market_depth_asks if side == 0 else self.market_depth_bids
        # if operation == 0 or operation == 1: # Insert or Update
        #     # position を使ってリストや順序付き辞書を更新する必要がある
        #     # ここでは単純化のため、価格をキーとしてサイズを格納 (順序は保証されない)
        #     target_book[price] = size
        # elif operation == 2: # Delete
        #     if price in target_book: # positionから正しい価格を見つける必要がある
        #         del target_book[price]
        
        self.mkt_depth_data_event.set() # 何かデータを受信したことを通知

    # updateMktDepthL2 も必要に応じて実装

def stream_market_depth(
    contract_symbol: str,
    contract_secType: str,
    contract_exchange: str,
    contract_currency: str,
    contract_lastTradeDateOrContractMonth: str = "", # 先物の場合に指定
    num_rows=10,
    duration_sec=30,
    host="127.0.0.1", port=7497, clientId=6
    ):
    """
    指定されたコントラクトの板情報をストリーミングする試み。
    動作にはマーケットデータ購読が必要。
    """
    fetcher = MarketDepthFetcher()
    print(f"Connecting to {host}:{port} with clientId:{clientId} for market depth...")
    fetcher.connect(host, port, clientId)

    if not fetcher.isConnected():
        print("Connection failed.")
        return

    api_thread = Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    if not fetcher.next_valid_id_event.wait(timeout=10):
        print("Timeout waiting for nextValidId.")
        fetcher.disconnect()
        api_thread.join()
        return

    contract = Contract()
    contract.symbol = contract_symbol
    contract.secType = contract_secType
    contract.exchange = contract_exchange
    contract.currency = contract_currency
    if contract_lastTradeDateOrContractMonth:
        contract.lastTradeDateOrContractMonth = contract_lastTradeDateOrContractMonth
    
    mkt_depth_req_id = fetcher.current_req_id + 1 # リクエストID
    is_smart_depth_request = False # reqMktDepthで使用した値を保持

    print(f"Requesting market depth for {contract.symbol} on {contract.exchange} (ReqId: {mkt_depth_req_id}, Rows: {num_rows}, SmartDepth: {is_smart_depth_request})...")
    fetcher.reqMktDepth(mkt_depth_req_id, contract, num_rows, is_smart_depth_request, [])

    print(f"Streaming market depth for {duration_sec} seconds...")
    # イベント発生またはタイムアウトまで待機
    # ここでは単純に時間で待つが、エラーイベントなども考慮すると良い
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        if fetcher.error_event.is_set():
            print("Error occurred, stopping market depth stream.")
            break
        time.sleep(0.1) # CPU負荷軽減

    print("Cancelling market depth subscription...")
    fetcher.cancelMktDepth(mkt_depth_req_id, is_smart_depth_request) # isSmartDepth引数を追加

    fetcher.disconnect()
    api_thread.join(timeout=5)
    print("Disconnected from IB for market depth.")

    # 取得した板情報を表示する処理 (fetcher.market_depth_bids, fetcher.market_depth_asks)
    # print("\nFinal Market Depth (simplified):")
    # print("Bids:", sorted(fetcher.market_depth_bids.items(), reverse=True))
    # print("Asks:", sorted(fetcher.market_depth_asks.items()))


if __name__ == '__main__':
    print("Attempting to stream N225 Futures market depth (conceptual)...")
    # !!! 以下のコントラクト情報はご自身の環境と購読に合わせてください !!!
    # N225先物の直近限月などを指定
    # 例: 2025年9月限
    
    # まずは限月リストを取得する (ib_futures_expirations.py を利用する想定)
    # from ib_futures_expirations import get_n225_futures_expirations
    # expirations = get_n225_futures_expirations(clientId=103) # clientIdは他と重複しないように
    # target_expiration_month = ""
    # if expirations:
    #     today_str = datetime.now().strftime("%Y%m%d")
    #     future_exp_dates = [d for d in expirations if d > today_str]
    #     if future_exp_dates:
    #         target_expiration_month = future_exp_dates[0][:6] # YYYYMM
    #         print(f"Targeting N225 Future month: {target_expiration_month}")
    #     else:
    #         print("No suitable future expiration month found.")
    # else:
    #     print("Could not get N225 futures expirations.")

    # 上記で取得した限月、または手動で設定
    target_expiration_month = "202509" # 例: 2025年9月限 (YYYYMM形式)
                                      # get_n225_futures_expirations から取得した YYYYMMDD の先頭6文字でも可

    if not target_expiration_month:
        print("Target expiration month not set. Exiting.")
    else:
        print(f"Note: Market depth data for N225 Futures on OSE.JPN requires a paid subscription.")
        stream_market_depth(
            contract_symbol="N225",
            contract_secType="FUT",
            contract_exchange="OSE.JPN",
            contract_currency="JPY",
            contract_lastTradeDateOrContractMonth=target_expiration_month,
            num_rows=5, # 表示する板の行数 (上下5行ずつ)
            duration_sec=20, # 20秒間ストリーミング
            clientId=104 # 他のAPIクライアントと重複しないように
        )