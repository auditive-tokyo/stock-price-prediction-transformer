from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper
from threading import Thread, Event
import time
from dotenv import load_dotenv
import os

load_dotenv()

IB_ACCOUNT = os.getenv("IB_ACCOUNT")

class AccountUpdateFetcher(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.done = Event()
        self.available_funds = None

    @iswrapper
    def updateAccountValue(self, key, value, currency, accountName):
        if key == "AvailableFunds" and currency == "JPY":
            self.available_funds = float(value)
        # print(f"updateAccountValue: {key}={value} {currency} ({accountName})")

    @iswrapper
    def accountDownloadEnd(self, accountName):
        print(f"accountDownloadEnd: {accountName}")
        self.done.set()

def get_account_updates(host="127.0.0.1", port=7497, clientId=999):
    fetcher = AccountUpdateFetcher()
    fetcher.connect(host, port, clientId)
    api_thread = Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    time.sleep(2)  # 接続後に少し待つ
    fetcher.reqAccountUpdates(True, IB_ACCOUNT)
    fetcher.done.wait(timeout=10)
    fetcher.disconnect()
    api_thread.join(timeout=5)

    # ここで保存した値を返す
    print(f"AvailableFunds for account {IB_ACCOUNT}: {fetcher.available_funds} JPY")
    return fetcher.available_funds

if __name__ == "__main__":
    funds = get_account_updates()
    print("AvailableFunds (JPY):", funds)