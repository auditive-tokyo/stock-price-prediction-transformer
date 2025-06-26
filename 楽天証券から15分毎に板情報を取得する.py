from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_board_info():
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)

    url = "https://member.rakuten-sec.co.jp/app/info_jp_prc_stock.do;BV_SessionID=4ACA473DA87A61A7DD293BE2794AB342.b756a4c7?eventType=init&type=&sub_type=&local=&searchType=0&dscrCd=7203&marketCd=1&gmn=J&smn=05&lmn=01&fmn=01"

    # すでに目的のページなら移動しない
    if driver.current_url != url:
        driver.get(url)

    # 板情報テーブルを取得
    rows = driver.find_elements(By.CSS_SELECTOR, "table.tbl-board tbody tr")
    board = []
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) < 5:
            continue  # 空行やヘッダー行はスキップ

        # 売数量
        sell_qty = cells[1].text.strip().replace(",", "")
        # 値段
        price = cells[2].text.strip().replace(",", "")
        # 買数量
        buy_qty = cells[4].text.strip().replace(",", "")

        # 空欄はNoneに
        sell_qty = int(sell_qty) if sell_qty else None
        price = price if price else None
        buy_qty = int(buy_qty) if buy_qty else None

        board.append({
            "sell_qty": sell_qty,
            "price": price,
            "buy_qty": buy_qty
        })

    return board

if __name__ == "__main__":
    board = get_board_info()
    for item in board:
        print(item)