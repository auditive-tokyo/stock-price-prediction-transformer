import time
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class ExpirationFetcher(EWrapper, EClient):
    """
    IB APIと通信して特定の先物の限月リストを取得するためのクラス。
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.contract_details_list = []
        self.all_expirations = []
        self.next_valid_id_event = Event()
        self.contract_details_end_event = Event()
        self.error_event = Event()
        self.error_message = ""
        self.current_req_id = 0

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_req_id = orderId # リクエストIDを保存
        self.next_valid_id_event.set() # nextValidId受信を通知

    @iswrapper
    def contractDetails(self, reqId: int, contractDetails):
        super().contractDetails(reqId, contractDetails)
        # print(f"  Contract Detail: {contractDetails.contract.symbol}, Month: {contractDetails.contract.lastTradeDateOrContractMonth}")
        self.contract_details_list.append(contractDetails.contract)

    @iswrapper
    def contractDetailsEnd(self, reqId: int):
        super().contractDetailsEnd(reqId)
        # print("ContractDetailsEnd received.")
        exp_months = set()
        for contract_item in self.contract_details_list:
            if contract_item.lastTradeDateOrContractMonth:
                exp_months.add(contract_item.lastTradeDateOrContractMonth)
        
        self.all_expirations = sorted(list(exp_months))
        self.contract_details_end_event.set() # contractDetailsEnd受信を通知

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # エラーコード200は「セキュリティ定義が見つからない」など、リクエストに問題がある場合
        # エラーコード2104, 2106, 2158などは接続ステータス通知なので無視してよい
        # 致命的なエラーのみを記録
        if errorCode not in [2104, 2105, 2106, 2107, 2108, 2158]:
            self.error_message = f"Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
            print(self.error_message) # デバッグ用にエラーを表示
            if errorCode == 200 and reqId == self.current_req_id: # 自分のリクエストに対するエラー
                self.error_event.set() # エラー発生を通知
            # 他の致命的なエラーでもイベントをセットするかは要件による

def get_n225_futures_expirations(base_contract: Contract, host: str, port: int, client_id: int):
    """
    Interactive Brokersから指定された基本コントラクトの取引可能な限月リストを取得

    Args:
        base_contract (Contract): 検索の基となるコントラクトオブジェクト (symbol, secType, exchange, currencyが設定されていること)。
        host (str): TWS/Gatewayのホスト名。
        port (int): TWS/Gatewayのポート番号。
        client_id (int): APIクライアントID。

    Returns:
        list: 取引可能な限月 (YYYYMMDD形式) のソート済みリスト。
              エラー発生時や取得できなかった場合は空リスト。
    """
    fetcher = ExpirationFetcher()
    
    print(f"Connecting to {host}:{port} with clientId:{client_id} for expirations...")
    fetcher.connect(host, port, client_id)

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

    # 引数で渡されたコントラクトオブジェクトを使用
    print(f"Requesting contract details for {base_contract.symbol} on {base_contract.exchange} with reqId {fetcher.current_req_id}...")
    fetcher.reqContractDetails(fetcher.current_req_id, base_contract)

    # contractDetailsEnd またはエラーの受信を待つ (タイムアウト付き)
    # イベントのいずれかがセットされるまで待機
    wait_start_time = time.time()
    timeout_seconds = 20 # タイムアウト時間を設定
    while not (fetcher.contract_details_end_event.is_set() or fetcher.error_event.is_set()):
        time.sleep(0.1) 
        if not api_thread.is_alive(): 
            print("API thread terminated unexpectedly.")
            break
        if time.time() - wait_start_time > timeout_seconds:
            print("Timeout waiting for contractDetailsEnd or error.")
            break


    expirations_list = []
    if fetcher.contract_details_end_event.is_set() and not fetcher.error_event.is_set():
        expirations_list = fetcher.all_expirations
        print(f"Successfully fetched {len(expirations_list)} expiration months.")
    elif fetcher.error_event.is_set():
        print(f"Failed to fetch expirations due to an error: {fetcher.error_message}")
    else:
        print("Failed to fetch expirations (timeout or unknown issue).")


    fetcher.disconnect()
    api_thread.join(timeout=5) 
    print("Disconnected from IB for expirations.")
    
    return expirations_list

# if __name__ == '__main__':
#     # このファイル単体で実行した場合のテスト用コード
#     print("Fetching N225 futures expirations as a standalone script...")

#     expirations = get_n225_futures_expirations(port=7497, clientId=99)
    
#     if expirations:
#         print("\nAvailable N225 Expiration Months on OSE.JPN:")
#         for month in expirations:
#             print(month)
#     else:
#         print("\nCould not retrieve N225 expiration months.")