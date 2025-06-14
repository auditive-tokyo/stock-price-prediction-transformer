import time
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.utils import iswrapper

class OrderPlacer(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_id_event = Event()
        self.order_status_event = Event()
        self.open_order_event = Event() # openOrderコールバック受信用
        self.error_event = Event()
        
        self.current_order_id = None
        self.order_feedback = {} # {orderId: {"status": str, "filled": float, ...}}
        self.error_message = ""

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_order_id = orderId
        print(f"OrderPlacer: nextValidId received: {orderId}")
        self.next_valid_id_event.set()

    @iswrapper
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice=0):
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        print(f"OrderPlacer: orderStatus - Id: {orderId}, Status: {status}, Filled: {filled}, Remaining: {remaining}, AvgFillPrice: {avgFillPrice}")
        if orderId == self.current_order_id:
            self.order_feedback[orderId] = {
                "status": status,
                "filled": filled,
                "remaining": remaining,
                "avgFillPrice": avgFillPrice
            }
            # "Submitted" や "Filled" など、重要なステータスでイベントをセット
            if status in ["Submitted", "Filled", "Cancelled", "ApiCancelled", "Inactive"]:
                self.order_status_event.set()
            if status == "Filled" and remaining == 0: # 完全約定
                print(f"Order {orderId} fully filled.")
                self.order_status_event.set()


    @iswrapper
    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        print(f"OrderPlacer: openOrder - Id: {orderId}, Status: {orderState.status}")
        # orderStatusと重複する情報もあるが、orderStateにはより詳細な情報が含まれることがある
        if orderId == self.current_order_id:
            if orderId not in self.order_feedback: # orderStatusより先に呼ばれる場合がある
                self.order_feedback[orderId] = {}
            self.order_feedback[orderId].update({
                "open_order_status": orderState.status,
                "commission": orderState.commission,
                "warningText": orderState.warningText
            })
            # openOrderでも主要なステータス変更を検知
            if orderState.status in ["Submitted", "Filled", "Cancelled", "ApiCancelled", "Inactive", "PendingSubmit", "PreSubmitted"]:
                 self.open_order_event.set()


    @iswrapper
    def execDetails(self, reqId, contract, execution):
        super().execDetails(reqId, contract, execution)
        print(f"OrderPlacer: execDetails - OrderId: {execution.orderId}, ExecId: {execution.execId}, AvgPrice: {execution.avgPrice}, CumQty: {execution.cumQty}")
        # 約定詳細。orderStatusのavgFillPriceやfilledと関連
        if execution.orderId == self.current_order_id:
            if execution.orderId not in self.order_feedback:
                self.order_feedback[execution.orderId] = {}
            self.order_feedback[execution.orderId].update({
                "last_exec_price": execution.price,
                "last_exec_qty": execution.shares, # 直近の約定数量
                "cum_qty": execution.cumQty,
                "avg_fill_price_exec": execution.avgPrice #約定ベースの平均価格
            })
            # 約定があればorder_status_eventをセットしても良い
            self.order_status_event.set()


    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # 接続関連の一般的な情報は無視 (2104, 2106, 2158など)
        # 注文関連のエラーコードは多数あるため、ここではreqIdが-1でないもの、または特定の注文IDに関連するものを重視
        if reqId == -1 and errorCode in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101]:
            print(f"OrderPlacer: Connection/Notification message: {errorCode} - {errorString}")
            return

        self.error_message = f"OrderPlacer Error. Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
        print(self.error_message)
        # 特定の注文IDに関連するエラーか、一般的な注文システムエラーか
        if self.current_order_id is not None and (reqId == self.current_order_id or reqId == -1):
            # reqIdが-1でも、内容が注文拒否などであればエラーとして扱う
            # (例: errorCode 201 - Order rejected - Reason: ...)
            # (例: errorCode 103 - Duplicate order id)
            # (例: errorCode 10147 - Order placement is rejected: BUY GTC MKT order would be rejected as it is not allowed.)
            if orderId := getattr(self, 'current_order_id', None): # current_order_id がセットされていれば
                 if orderId not in self.order_feedback: self.order_feedback[orderId] = {}
                 self.order_feedback[orderId]["status"] = "Error"
                 self.order_feedback[orderId]["error_message"] = errorString
                 self.order_feedback[orderId]["error_code"] = errorCode
            self.error_event.set()
            self.order_status_event.set() # エラー時もイベントを発生させて待機を解除
            self.open_order_event.set()


def place_n225_market_order(
    action: str, # "BUY" or "SELL"
    quantity: int,
    expiration_month: str, # YYYYMM format, e.g., "202509"
    host="127.0.0.1", port=7497, clientId=7 # clientIdは他と重複しないように
    ):
    """
    N225先物に対して成り行き注文を発注します。

    Args:
        action (str): "BUY" または "SELL"。
        quantity (int): 発注数量。
        expiration_month (str): 限月 (YYYYMM形式)。
        host (str): TWS/Gatewayのホスト。
        port (int): TWS/Gatewayのポート。
        clientId (int): APIクライアントID。

    Returns:
        dict: 注文結果のフィードバック。
              例: {"status": "Submitted", "orderId": 123, "message": "Order submitted."}
                  {"status": "Error", "message": "Error details..."}
    """
    if action.upper() not in ["BUY", "SELL"]:
        return {"status": "Error", "message": "Invalid action. Must be 'BUY' or 'SELL'."}
    if quantity <= 0:
        return {"status": "Error", "message": "Quantity must be positive."}

    placer = OrderPlacer()
    
    print(f"Connecting to {host}:{port} with clientId:{clientId} for order placement...")
    placer.connect(host, port, clientId)

    if not placer.isConnected():
        print("Connection failed.")
        return {"status": "Error", "message": "Connection to TWS/Gateway failed."}

    api_thread = Thread(target=placer.run, daemon=True)
    api_thread.start()

    print("Waiting for nextValidId...")
    if not placer.next_valid_id_event.wait(timeout=10):
        print("Timeout waiting for nextValidId.")
        placer.disconnect()
        api_thread.join()
        return {"status": "Error", "message": "Timeout waiting for nextValidId from TWS/Gateway."}
    
    order_id = placer.current_order_id
    placer.order_feedback[order_id] = {} # 初期化

    # --- Contractの定義 ---
    contract = Contract()
    contract.symbol = "N225"
    contract.secType = "FUT"
    contract.exchange = "OSE.JPN" # 大阪取引所
    contract.currency = "JPY"
    contract.lastTradeDateOrContractMonth = expiration_month

    # --- Orderの定義 ---
    order = Order()
    order.action = action.upper()
    order.orderType = "MKT" # 成り行き注文
    order.totalQuantity = quantity
    order.transmit = True # 即時送信

    # --- EtradeOnlyエラー対策として関連しそうな属性を明示的に初期化 ---
    # これらの属性が直接の原因かは不明ですが、不要なフラグをクリアする試み
    order.eTradeOnly = False  # 明示的にFalse (標準属性ではないが念のため)
    order.firmQuoteOnly = False
    # order.optOutSmartRouting = False # OSE.JPN直指定なので通常不要
    # order.overridePercentageConstraints = False # 通常のMKT注文では不要

    # その他の属性も必要に応じてデフォルト値に設定
    # order.account = "" # 通常はAPI接続時のアカウントが使われる
    # order.tif = "DAY" # Time in Force, MKTなら通常DAYだが明示しても良い

    print(f"Placing {action} MKT order for {quantity} of N225 {expiration_month} (OrderId: {order_id})...")
    placer.placeOrder(order_id, contract, order)

    # 注文ステータスまたはエラーの受信を待機 (タイムアウト付き)
    # orderStatus, openOrder, error のいずれかでイベントがセットされるのを待つ
    # 複数のイベントを監視するために any_event のような仕組みがあると良いが、ここでは単純化
    timeout_seconds = 15 # ステータス確認のタイムアウト
    print(f"Waiting for order status confirmation (up to {timeout_seconds}s)...")
    
    # イベントのいずれかが発生するのを待つ
    # Python 3.x: threading.Event.wait() returns True if the internal flag is true upon calling wait, False if the timeout occurs.
    # We need to check multiple events. A simple way is to loop or use a master event.
    # For simplicity, we'll wait on order_status_event which should be set by orderStatus or error.
    
    wait_start_time = time.time()
    final_feedback = {"status": "Unknown", "orderId": order_id, "message": "No definitive status received within timeout."}

    while time.time() - wait_start_time < timeout_seconds:
        if placer.error_event.is_set(): # まずエラーを確認
            feedback = placer.order_feedback.get(order_id, {})
            final_feedback = {
                "status": "Error",
                "orderId": order_id,
                "message": feedback.get("error_message", placer.error_message or "An error occurred."),
                "errorCode": feedback.get("error_code", "N/A")
            }
            break
        if placer.order_status_event.is_set() or placer.open_order_event.is_set():
            feedback = placer.order_feedback.get(order_id, {})
            current_status = feedback.get("status") or feedback.get("open_order_status")
            if current_status:
                final_feedback = {
                    "status": current_status,
                    "orderId": order_id,
                    "filled": feedback.get("filled", 0),
                    "remaining": feedback.get("remaining", quantity),
                    "avgFillPrice": feedback.get("avgFillPrice", 0.0),
                    "message": f"Order status: {current_status}"
                }
                if current_status in ["Submitted", "Filled", "PendingSubmit", "PreSubmitted"]:
                    break # これらのステータスなら一旦完了
                elif current_status in ["Cancelled", "ApiCancelled", "Inactive"]:
                    final_feedback["message"] = f"Order ended with status: {current_status}"
                    break
            # イベントがセットされてもフィードバックがまだない場合、少し待つ
        time.sleep(0.2)
    else: # ループがタイムアウトで終了した場合
        # タイムアウト時点での最新のフィードバックを取得試行
        feedback = placer.order_feedback.get(order_id, {})
        current_status = feedback.get("status") or feedback.get("open_order_status")
        if current_status:
             final_feedback = {
                "status": current_status or "TimeoutNoStatus",
                "orderId": order_id,
                "filled": feedback.get("filled", 0),
                "remaining": feedback.get("remaining", quantity),
                "avgFillPrice": feedback.get("avgFillPrice", 0.0),
                "message": f"Order status after timeout: {current_status or 'No specific status update received.'}"
            }

    placer.disconnect()
    api_thread.join(timeout=5)
    print(f"Disconnected. Final order feedback for {order_id}: {final_feedback}")
    return final_feedback

if __name__ == '__main__':
    print("--- Market Order Placement Test ---")
    # !!! ペーパートレーディングアカウントで十分にテストしてください !!!
    # !!! 実際の資金をリスクに晒す可能性があります !!!

    # --- 設定 ---
    target_client_id = 201 # 他のスクリプトと重複しないように！
    target_port = 7497      # TWSペーパーアカウントのデフォルトポート
    
    # N225先物の限月 (YYYYMM形式)
    # ib_futures_expirations.py などで動的に取得することを推奨
    n225_expiration = "202509" # 例: 2025年9月限 (実際の取引可能な限月にしてください)
    
    order_qty = 1 # 発注枚数

    # --- 買い注文のテスト ---
    print(f"\nAttempting to place BUY order for N225 {n225_expiration}, Qty: {order_qty}")
    buy_result = place_n225_market_order(
        action="SELL",
        quantity=order_qty,
        expiration_month=n225_expiration,
        clientId=target_client_id,
        port=target_port
    )
    print(f"BUY Order Result: {buy_result}")

    # # --- 売り注文のテスト (注意して実行) ---
    # # ポジションがない場合に売り注文を出すと新規ショートポジションになります。
    # time.sleep(5) # APIのクールダウンや次の注文ID取得のため少し待つ
    # target_client_id +=1 # 次の注文のためにclientIdを変えるか、同じクライアントでnextValidIdを待つ
    # print(f"\nAttempting to place SELL order for N225 {n225_expiration}, Qty: {order_qty}")
    # sell_result = place_n225_market_order(
    #     action="SELL",
    #     quantity=order_qty,
    #     expiration_month=n225_expiration,
    #     clientId=target_client_id + 1, # 前のテストとIDを変える
    #     port=target_port
    # )
    # print(f"SELL Order Result: {sell_result}")

    print("\n--- Test Finished ---")
    print("IMPORTANT: Check your TWS/Gateway for actual order status and executions.")