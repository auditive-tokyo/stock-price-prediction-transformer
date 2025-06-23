import time
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class SnapshotQuoteFetcher(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_id_event = Event()
        self.snapshot_end_event = Event()
        self.error_message = ""
        self.current_req_id = 0
        self.last_price = None

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.current_req_id = orderId
        self.next_valid_id_event.set()

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]: return
        self.error_message = f"API Error: Id: {reqId}, Code: {errorCode}, Msg: {errorString}"
        print(self.error_message)
        self.snapshot_end_event.set()

    @iswrapper
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        if reqId == self.current_req_id and tickType == 4: # 4 = LAST_PRICE
            self.last_price = price

    @iswrapper
    def tickSnapshotEnd(self, reqId: int):
        super().tickSnapshotEnd(reqId)
        if reqId == self.current_req_id:
            print("Tick snapshot end received.")
            self.snapshot_end_event.set()

def get_us_stock_snapshot(host="127.0.0.1", port=4001, clientId=109):
    fetcher = SnapshotQuoteFetcher()
    fetcher.connect(host, port, clientId)
    api_thread = Thread(target=fetcher.run, daemon=True); api_thread.start()

    if not fetcher.next_valid_id_event.wait(timeout=10):
        print("Timeout waiting for nextValidId."); fetcher.disconnect(); return

    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    req_id = fetcher.current_req_id + 1
    fetcher.current_req_id = req_id
    
    print(f"Requesting snapshot for {contract.symbol}...")
    fetcher.reqMktData(req_id, contract, "", True, False, [])

    event_triggered = fetcher.snapshot_end_event.wait(timeout=15)
    fetcher.disconnect()

    if event_triggered and fetcher.last_price is not None:
        print(f"\nSuccess! Last price for {contract.symbol}: {fetcher.last_price}")
    elif fetcher.error_message:
        print(f"\nFailed with error: {fetcher.error_message}")
    else:
        print("\nFailed: Timeout waiting for snapshot data.")

if __name__ == '__main__':
    print("--- US Stock Snapshot Test ---")
    get_us_stock_snapshot(port=4001, clientId=110)