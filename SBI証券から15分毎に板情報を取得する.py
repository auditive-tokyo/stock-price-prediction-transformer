import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

LOGIN_ID = os.getenv("SBI_LOGIN_ID")
LOGIN_PASSWORD = os.getenv("SBI_LOGIN_PASSWORD")

def get_board_info():
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=chrome_options)

    login_url = "https://site3.sbisec.co.jp/ETGate"
    driver.get(login_url)

    wait = WebDriverWait(driver, 10)
    # ユーザー名入力
    user_input = wait.until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[5]/div[4]/form/div[1]/input"))
    )
    user_input.clear()
    user_input.send_keys(LOGIN_ID)
    time.sleep(1)

    # パスワード入力
    password_input = wait.until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[5]/div[4]/form/div[2]/div/input"))
    )
    password_input.clear()
    password_input.send_keys(LOGIN_PASSWORD)
    time.sleep(1)

    # ログインボタン（submit）をクリック
    submit_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div[5]/div[4]/form/p[2]/input"))
    )
    submit_button.click()

    input("ログイン後、Enterキーを押してください...")

    url = "https://site3.sbisec.co.jp/"
    if driver.current_url != url:
        driver.get(url)
    time.sleep(2)

    # 「先物・オプション」リンクをクリック
    wait = WebDriverWait(driver, 10)
    futures_link = wait.until(
        EC.element_to_be_clickable((By.ID, "futures-global-nav"))
    )
    futures_link.click()
    time.sleep(2)

    # 「先物・オプション取引サイト」ボタンのリンクを取得
    button_group = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.seeds-button-group"))
    )
    link_elem = button_group.find_element(By.TAG_NAME, "a")
    url_2 = link_elem.get_attribute("href")
    print(f"url_2: {url_2}")

    if driver.current_url != url_2:
        driver.get(url_2)
    time.sleep(2)

    # 「取引」ページに移管
    url_3 = "https://jderivatives.sbisec.co.jp/main/trade/future"
    if driver.current_url != url_3:
        driver.get(url_3)
    time.sleep(2)

    # 「日経225先物」ラジオボタンをクリック
    wait = WebDriverWait(driver, 10)
    n225_radio_label = wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//label[contains(@class, 'ant-radio-wrapper') and contains(., '日経225先物')]")
        )
    )
    n225_radio_input = n225_radio_label.find_element(By.CSS_SELECTOR, "input.ant-radio-input")
    driver.execute_script("arguments[0].scrollIntoView(true);", n225_radio_input)
    n225_radio_input.click()
    time.sleep(2)

    # 板情報テーブルが現れるまで待機
    wait = WebDriverWait(driver, 10)
    table = wait.until(
        EC.presence_of_element_located((
            By.XPATH,
            "/html/body/app-root/div/nz-spin/div/oms-main/section/div[4]/as-split/as-split-area[2]/div/oms-new-future-order/section/div[2]/oms-quotation-board/section/div[3]/div[1]/table"
        ))
    )

    # --- ここからスイッチをオンにする処理を追加 ---
    switch = wait.until(
        EC.element_to_be_clickable((
            By.XPATH,
            "/html/body/app-root/div/nz-spin/div/oms-main/section/div[4]/as-split/as-split-area[2]/div/oms-new-future-order/section/div[2]/oms-quotation-board/section/div[1]/div[2]/nz-switch"
        ))
    )

    # スイッチのbutton要素を取得
    switch_button = switch.find_element(By.TAG_NAME, "button")
    switch_class = switch_button.get_attribute("class")

    if "ant-switch-checked" not in switch_class:
        # オフなのでクリックしてオンにする
        switch_button.click()
    else:
        print("自動更新スイッチはすでにオンです")

    # board_infoディレクトリの作成
    board_info_dir = os.path.join(os.path.dirname(__file__), "board_info")
    os.makedirs(board_info_dir, exist_ok=True)

    board = []
    while True:
        table = driver.find_element(By.XPATH, "/html/body/app-root/div/nz-spin/div/oms-main/section/div[4]/as-split/as-split-area[2]/div/oms-new-future-order/section/div[2]/oms-quotation-board/section/div[3]/div[1]/table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        board.clear()
        for i, row in enumerate(rows[1:]):  # 1行目はヘッダーなのでスキップ
            cells = row.find_elements(By.TAG_NAME, "td")

            if len(cells) < 3:
                print(f"  → スキップ（セル数{len(cells)}）")
                continue

            sell_qty_raw = cells[0].text.strip().replace(",", "")
            if sell_qty_raw and not sell_qty_raw.isdigit():
                print(f"  → スキップ（1列目: {sell_qty_raw}）")
                continue

            price_raw = cells[1].text.strip().replace(",", "")
            buy_qty_raw = cells[2].text.strip().replace(",", "")

            sell_qty = int(sell_qty_raw) if sell_qty_raw.isdigit() else None
            price = price_raw if price_raw else None
            buy_qty = int(buy_qty_raw) if buy_qty_raw.isdigit() else None

            board.append({
                "sell_qty": sell_qty,
                "price": price,
                "buy_qty": buy_qty
            })

        # ここでboardをjson保存（毎回上書き）
        json_path = os.path.join(board_info_dir, "board_info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(board, f, ensure_ascii=False, indent=2)

        print(f"15分ごとに板情報を取得し、{json_path} に保存しました")
        time.sleep(15 * 60)

if __name__ == "__main__":
    board = get_board_info()
