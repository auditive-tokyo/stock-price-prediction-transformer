import time
from threading import Thread, Event
import csv

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class IBapi(EWrapper, EClient):
    def __init__(self, base_contract_details: Contract, contract_month_to_request: str, bar_size_setting: str, duration_string: str):
        EClient.__init__(self, self)
        self.data = [["Date", "Open", "High", "Low", "Close", "Volume"]]
        self.nextOrderId = None
        self.base_contract = base_contract_details
        self.contract_month = contract_month_to_request
        self.bar_size = bar_size_setting # barSizeSettingを保持
        self.duration = duration_string   # durationStrを保持
        self.historical_data_end_event = Event()

    @iswrapper
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextOrderId = orderId
        print(f"nextValidId (for historical data request): {orderId}, BarSize: {self.bar_size}, Duration: {self.duration}")
        self.request_historical_data()

    @iswrapper
    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print(f"HistoricalDataEnd. ReqId: {reqId} from {start} to {end} for BarSize: {self.bar_size}")
        self.historical_data_end_event.set()

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2105, 2106, 2107, 2108, 2158, 2100, 2101, 2119, 10167, 162]: # 162: Historical Market Data Service error message
            # 162はデータがない場合などにも出るので、INFO扱いにするか検討
            print(f"INFO/DEBUG Message (Historical Data): Id: {reqId}, Code: {errorCode}, Msg: {errorString}, BarSize: {self.bar_size if hasattr(self, 'bar_size') else 'N/A'}")
            # If it's a "no data" error for a specific request, we might still want to signal completion.
            if errorCode == 162 and "HMDS query returned no data" in errorString:
                self.historical_data_end_event.set() # データなしでも完了として扱う
            return
        log_message = f"Error (Historical Data). Id: {reqId}, Code: {errorCode}, Msg: {errorString}, BarSize: {self.bar_size if hasattr(self, 'bar_size') else 'N/A'}"
        if advancedOrderRejectJson and advancedOrderRejectJson != "":
            log_message += f", AdvancedOrderReject: {advancedOrderRejectJson}"
        print(log_message)
        # Consider setting historical_data_end_event on critical errors to unblock waiting
        # if errorCode not in [SOME_NON_CRITICAL_ERROR_CODES_FOR_HISTORICAL_DATA]:
        #     self.historical_data_end_event.set()


    def request_historical_data(self):
        contract_to_fetch = Contract()
        contract_to_fetch.symbol = self.base_contract.symbol
        contract_to_fetch.secType = self.base_contract.secType
        contract_to_fetch.exchange = self.base_contract.exchange
        contract_to_fetch.currency = self.base_contract.currency
        if self.base_contract.tradingClass:
             contract_to_fetch.tradingClass = self.base_contract.tradingClass
        contract_to_fetch.lastTradeDateOrContractMonth = self.contract_month

        req_id_for_hist_data = self.nextOrderId if self.nextOrderId is not None else 1

        print(f"Requesting historical data for {contract_to_fetch.symbol} {contract_to_fetch.lastTradeDateOrContractMonth} on {contract_to_fetch.exchange} with ReqId: {req_id_for_hist_data}...")
        print(f"Duration: {self.duration}, Bar Size: {self.bar_size}")
        self.reqHistoricalData(
            reqId=req_id_for_hist_data,
            contract=contract_to_fetch,
            endDateTime="",
            durationStr=self.duration,       # 引数で渡された期間を使用
            barSizeSetting=self.bar_size,    # 引数で渡された足種を使用
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

def run_loop(app_instance):
    app_instance.run()
