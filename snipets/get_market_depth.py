import time
from threading import Thread, Event
import os
from datetime import datetime

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
        # 板情報を格納するためのリスト (順序が重要)
        self.market_depth_bids = [] # [(price, size), ...]
        self.market_depth_asks = [] # [(price, size), ...]

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_req_id = orderId
        self.next_valid_id_event.set()

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # 接続関連の一般的な情報は無視
        if errorCode in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101, 2119]:
            return
        self.error_message = f"MktDepth Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
        print(self.error_message)
        self.error_event.set()

    @iswrapper
    def updateMktDepth(self, reqId: int, position: int, operation: int, side: int, price: float, size: int):
        super().updateMktDepth(reqId, position, operation, side, price, size)
        
        target_book = self.market_depth_asks if side == 0 else self.market_depth_bids

        # operation: 0 = Insert, 1 = Update, 2 = Delete
        if operation == 0: # Insert
            target_book.insert(position, (price, size))
        elif operation == 1: # Update
            if position < len(target_book):
                target_book[position] = (price, size)
        elif operation == 2: # Delete
            if position < len(target_book):
                target_book.pop(position)

    def print_market_depth(self):
        """現在の板情報を整形して表示する"""
        os.system('cls' if os.name == 'nt' else 'clear') # 画面をクリア
        print(f"--- Market Depth (Top 5) as of {datetime.now().strftime('%H:%M:%S')} ---")
        print("         BIDS (買い)      |      ASKS (売り)")
        print("--------------------------|--------------------------")
        print("   Price    |    Size     |    Price    |    Size")
        print("--------------------------|--------------------------")

        for i in range(5):
            bid_price, bid_size = self.market_depth_bids[i] if i < len(self.market_depth_bids) else ("-", "-")
            ask_price, ask_size = self.market_depth_asks[i] if i < len(self.market_depth_asks) else ("-", "-")
            
            bid_price_str = f"{bid_price:.2f}" if isinstance(bid_price, float) else bid_price
            ask_price_str = f"{ask_price:.2f}" if isinstance(ask_price, float) else ask_price

            print(f" {bid_price_str:>10} | {bid_size:>10} | {ask_price_str:>10} | {ask_size:>10}")
        print("------------------------------------------------------")


def stream_market_depth(
    contract_symbol: str,
    contract_secType: str,
    contract_exchange: str,
    contract_currency: str,
    contract_lastTradeDateOrContractMonth: str = "",
    num_rows=5,
    duration_sec=30,
    host="127.0.0.1", port=4001, clientId=105
    ):
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
        return

    contract = Contract()
    contract.symbol = contract_symbol
    contract.secType = contract_secType
    contract.exchange = contract_exchange
    contract.currency = contract_currency
    if contract_lastTradeDateOrContractMonth:
        contract.lastTradeDateOrContractMonth = contract_lastTradeDateOrContractMonth
    
    mkt_depth_req_id = fetcher.current_req_id + 1
    is_smart_depth_request = False

    print(f"Requesting market depth for {contract.symbol} on {contract.exchange}...")
    fetcher.reqMktDepth(mkt_depth_req_id, contract, num_rows, is_smart_depth_request, [])

    print(f"Streaming market depth for {duration_sec} seconds... (Press Ctrl+C to stop early)")
    start_time = time.time()
    try:
        while time.time() - start_time < duration_sec:
            if fetcher.error_event.is_set():
                print("An error occurred, stopping market depth stream.")
                break
            fetcher.print_market_depth()
            time.sleep(1) # 1秒ごとに画面を更新
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("Cancelling market depth subscription...")
    fetcher.cancelMktDepth(mkt_depth_req_id, is_smart_depth_request)

    fetcher.disconnect()
    print("Disconnected from IB for market depth.")


if __name__ == '__main__':
    # 本番ポートと、他と重複しないクライアントIDを使用
    target_port = 4001 
    target_client_id = 110

    # main.pyなどから取得したアクティブな限月を使用するのが望ましい
    # ここでは手動で設定
    target_expiration_month = "202509" 

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
            num_rows=5,
            duration_sec=60,
            port=target_port,
            clientId=target_client_id
        )