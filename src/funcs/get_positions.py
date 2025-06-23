import time
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class PositionFetcher(EWrapper, EClient):
    """
    IB APIと通信して現在のポートフォリオのポジションを取得するためのクラス。
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.positions = []
        self.position_end_event = Event()
        self.error_message = ""

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # 接続関連の一般的な情報は無視
        if errorCode in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101]:
            # print(f"PositionFetcher: Connection/Notification message: {errorCode} - {errorString}")
            return
        
        self.error_message = f"PositionFetcher Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
        print(self.error_message)
        # エラーが発生したら、待機を解除するためにイベントをセットする
        self.position_end_event.set()

    @iswrapper
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        super().position(account, contract, position, avgCost)
        # ポジションが0のものは表示しない
        if position != 0:
            # 先物の場合、avgCostは price * multiplier で返されることがあるため、
            # 建玉単価 (avgPrice) を計算します。
            # 正確なmultiplierはreqContractDetailsで取得するのが望ましいですが、
            # ここではシンボルに基づいて簡易的に判定します。
            multiplier = 1
            if contract.secType == "FUT":
                if contract.symbol == "N225":
                    multiplier = 1000  # 日経225先物ラージ
                elif contract.symbol == "N225M":
                    multiplier = 100  # 日経225先物ミニ

            avg_price = avgCost
            if contract.secType == "FUT" and multiplier > 1:
                # 先物の場合、avgCostは1単位あたりの価値（価格×乗数）として返されることが多いため、
                # avgCostを乗数で割って平均価格を算出します。
                avg_price = avgCost / multiplier

            self.positions.append({
                "account": account,
                "symbol": contract.symbol,
                "secType": contract.secType,
                "exchange": contract.exchange,
                "currency": contract.currency,
                "lastTradeDateOrContractMonth": contract.lastTradeDateOrContractMonth,
                "position": position,
                "avgCost": avgCost, # APIから返される元のavgCost（1単位あたりの価値）
                "avgPrice": avg_price # 計算された平均建玉単価
            })
        
    @iswrapper
    def positionEnd(self):
        super().positionEnd()
        print("PositionEnd received.")
        self.position_end_event.set()

def get_current_positions(host="127.0.0.1", port=7497, clientId=8):
    """
    TWS/Gatewayに接続し、現在の全ポジションを取得する。

    Args:
        host (str): TWS/Gatewayのホスト。
        port (int): TWS/Gatewayのポート。
        clientId (int): APIクライアントID。

    Returns:
        list or None: ポジション情報の辞書のリスト。エラーの場合はNoneを返す。
    """
    fetcher = PositionFetcher()

    print(f"Connecting to {host}:{port} with clientId:{clientId} for position data...")
    fetcher.connect(host, port, clientId)

    if not fetcher.isConnected():
        print("Connection failed.")
        return None

    api_thread = Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    time.sleep(1) # 接続が確立されるまで少し待つ

    print("Requesting current positions...")
    fetcher.reqPositions()

    timeout_seconds = 10
    print(f"Waiting for position data (up to {timeout_seconds}s)...")
    
    event_triggered = fetcher.position_end_event.wait(timeout=timeout_seconds)

    fetcher.cancelPositions() # リクエストをキャンセル
    fetcher.disconnect()
    api_thread.join(timeout=5)
    print("Disconnected.")

    if not event_triggered:
        print("Timeout waiting for position data.")
        return None

    if fetcher.error_message:
        print(f"An error occurred: {fetcher.error_message}")
        return None

    return fetcher.positions

if __name__ == '__main__':
    # このファイル単体で実行した場合のテスト用コード
    print("--- Get Current Positions Test ---")
    
    target_client_id = 202 # 他のスクリプトと重複しないように設定
    target_port = 7497

    positions = get_current_positions(port=target_port, clientId=target_client_id)

    if positions is None:
        print("\nCould not retrieve positions.")
    elif not positions:
        print("\nNo open positions found.")
    else:
        print("\n--- Current Positions ---")
        for pos in positions:
            print(f"  Symbol: {pos['symbol']}, SecType: {pos['secType']}, Expiry: {pos['lastTradeDateOrContractMonth']}, "
                  f"Qty: {pos['position']}, AvgPrice: {pos['avgPrice']:.2f}")
        print("-------------------------")

    print("\n--- Test Finished ---")