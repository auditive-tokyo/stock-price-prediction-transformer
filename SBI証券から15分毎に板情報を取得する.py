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
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
import random
from selenium.webdriver.common.action_chains import ActionChains
from main import N225_FUTURES_CONTRACT

load_dotenv()

LOGIN_ID = os.getenv("SBI_LOGIN_ID")
LOGIN_PASSWORD = os.getenv("SBI_LOGIN_PASSWORD")

def get_board_info(driver, relogin=False):
    print("=== SBI証券 板情報取得 開始 ===")
    login_url = "https://site3.sbisec.co.jp/ETGate"
    print("STEP 1: ログインページへ遷移")
    driver.get(login_url)

    wait = WebDriverWait(driver, 10)
    print("STEP 2: ユーザーID入力欄取得")
    user_input = wait.until(
        EC.presence_of_element_located((By.NAME, "user_id"))
    )
    user_input.clear()
    user_input.send_keys(LOGIN_ID)
    time.sleep(1 + random.uniform(0, 1))

    print("STEP 3: パスワード入力欄取得")
    password_input = wait.until(
        EC.presence_of_element_located((By.NAME, "user_password"))
    )
    password_input.clear()
    password_input.send_keys(LOGIN_PASSWORD)
    time.sleep(1 + random.uniform(0, 1))

    print("STEP 4: ログインボタン取得・クリック")
    submit_button = wait.until(
        EC.element_to_be_clickable((By.NAME, "ACT_login"))
    )
    submit_button.click()

    if not relogin:
        input("ログイン後、Enterキーを押してください...")

    # url = "https://site3.sbisec.co.jp/"
    # if driver.current_url != url:
    #     driver.get(url)
    # time.sleep(2 + random.uniform(0, 1))

    # 「先物・オプション」リンクをクリック
    print("STEP 5: 先物・オプションリンククリック")
    wait = WebDriverWait(driver, 10)
    futures_link = wait.until(
        EC.element_to_be_clickable((By.ID, "futures-global-nav"))
    )
    futures_link.click()
    time.sleep(2 + random.uniform(0, 1))

    print("STEP 6: 先物・オプション取引サイトボタン取得")
    button_group = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.seeds-button-group"))
    )
    link_elem = button_group.find_element(By.TAG_NAME, "a")
    url_2 = link_elem.get_attribute("href")
    print(f"url_2: {url_2}")

    if driver.current_url != url_2:
        print("STEP 7: 取引サイトへ遷移")
        driver.get(url_2)
    time.sleep(2 + random.uniform(0, 1))

    url_3 = "https://jderivatives.sbisec.co.jp/main/trade/future"
    if driver.current_url != url_3:
        print("STEP 8: 取引ページへ遷移")
        driver.get(url_3)
    time.sleep(2 + random.uniform(0, 1))

    print("STEP 9: シンボル選択")
    print("シンボルの状態:", N225_FUTURES_CONTRACT.symbol)
    if N225_FUTURES_CONTRACT.symbol == "N225":
        label_text = "日経225先物"
    elif N225_FUTURES_CONTRACT.symbol == "N225M":
        label_text = "ミニ日経225先物"
    else:
        label_text = "日経225先物"

    radio_span_xpath = f"//label[contains(@class, 'ant-radio-wrapper')]/span[2][normalize-space(text())='{label_text}']"
    wait = WebDriverWait(driver, 10)
    print(f"STEP 10: ラジオボタン選択: {label_text}")
    radio_span = wait.until(
        EC.element_to_be_clickable((By.XPATH, radio_span_xpath))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", radio_span)
    radio_span.click()
    time.sleep(2 + random.uniform(0, 1))

    print("STEP 11: 板情報テーブル取得")
    tables = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.no-table-bg"))
    )
    table = None
    for t in tables:
        headers = [th.text.strip() for th in t.find_elements(By.TAG_NAME, "th")]
        if headers[:3] == ["売気配", "値段", "買気配"]:
            table = t
            break
    if table is None:
        print("ERROR: 板情報テーブルが見つかりませんでした")
        raise NoSuchElementException("板情報テーブルが見つかりませんでした")
    print("STEP 11: 板情報テーブル取得 成功")

    print("STEP 12: 自動更新スイッチ取得")
    switch = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//nz-switch[contains(@class, 'auto')]"))
    )
    switch_button = switch.find_element(By.TAG_NAME, "button")
    switch_class = switch_button.get_attribute("class")

    if "ant-switch-checked" not in switch_class:
        print("STEP 13: 自動更新スイッチON")
        switch_button.click()
    else:
        print("STEP 13: 自動更新スイッチはすでにオン")

    board_info_dir = os.path.join(os.path.dirname(__file__), "board_info")
    os.makedirs(board_info_dir, exist_ok=True)

    board = []
    print("=== 板情報取得ループ開始 ===")
    while True:
        try:
            print("LOOP: 板情報テーブル再取得")
            tables = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.no-table-bg"))
            )
            table = None
            for t in tables:
                headers = [th.text.strip() for th in t.find_elements(By.TAG_NAME, "th")]
                if headers[:3] == ["売気配", "値段", "買気配"]:
                    table = t
                    break
            if table is None:
                print("ERROR: 板情報テーブルが見つかりませんでした（ループ内）")
                raise NoSuchElementException("板情報テーブルが見つかりませんでした")

            rows = table.find_elements(By.TAG_NAME, "tr")
            board.clear()
            for i, row in enumerate(rows[1:]):
                try:
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
                except StaleElementReferenceException:
                    print("StaleElementReferenceException: セル取得時に要素が無効化されました。テーブル再取得してリトライします。")
                    break
            else:
                json_path = os.path.join(board_info_dir, "board_info.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(board, f, ensure_ascii=False, indent=2)
                print(f"15分ごとに板情報を取得し、{json_path} に保存しました")
                time.sleep(15 * 60)
                continue

        except (NoSuchElementException, TimeoutException):
            print("セッション切れを検知。再ログインします。")
            get_board_info(driver, relogin=True)
            return
        except StaleElementReferenceException:
            print("StaleElementReferenceException: テーブル取得時に要素が無効化されました。再取得します。")
            continue

if __name__ == "__main__":
    chrome_options = Options()
    # chrome_options.add_experimental_option("detach", True)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP', 'ja']});
            """
        },
    )

    get_board_info(driver, relogin=False)
