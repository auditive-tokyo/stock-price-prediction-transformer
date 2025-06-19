import pandas as pd
import os
import datetime
from ibapi.contract import Contract
from .create_csv4charts import IBapi, run_loop

from threading import Thread, Event
import time
import csv

def parse_datetime_flexible(date_str):
    """
    'YYYYMMDD HH:MM:SS' または 'YYYYMMDD' 形式の文字列をdatetimeオブジェクトにパースする。
    パースできない場合は None を返す。
    """
    formats_to_try = ['%Y%m%d %H:%M:%S', '%Y%m%d']
    for fmt in formats_to_try:
        try:
            return datetime.datetime.strptime(str(date_str).strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None

def calculate_duration_since(last_date_str: str, bar_size: str) -> str:
    """
    指定された最終日時文字列から現在までの期間をIBKR APIのdurationStr形式で計算する。
    """
    last_dt = parse_datetime_flexible(last_date_str)
    if last_dt is None:
        print(f"Warning: Could not parse last_date_str: {last_date_str}. Defaulting to '1 W'.")
        return "1 W" # パース失敗時は広めに取る

    now = datetime.datetime.now()
    delta = now - last_dt

    # last_dt が日付のみ (時刻が00:00:00) の場合、その日の終わりまで考慮するイメージで調整
    # ただし、IBKRのdurationはそこまで細かくないので、基本は日数でカバー
    if last_dt.hour == 0 and last_dt.minute == 0 and last_dt.second == 0:
        # 日付のみの場合、delta.days が0でも当日分は取得したいので、最小1Dとする
        days_to_request = max(delta.days, 0) + 1 # 少なくとも当日分 + 余裕分
    else:
        days_to_request = delta.days + 1 # 最終取引時刻からなので、当日分 + 余裕分

    # 安全マージンとしてさらに1日加える（週末やAPIの挙動を考慮）
    days_to_request +=1

    if days_to_request <= 1: # 数時間以内の差分の場合
        if "min" in bar_size.lower():
            return "1 D" # 15分足などでも、差分更新なら1日分取れば十分なことが多い
        elif "hour" in bar_size.lower():
            return "2 D" # 時間足なら2日分
        else: # 日足
            return f"{max(days_to_request, 2)} D" # 日足でも最低2日分
    elif days_to_request <= 7:
        return f"{days_to_request} D"
    elif days_to_request <= 30 : # 約1ヶ月以内
        weeks_to_request = (days_to_request // 7) + 1
        return f"{weeks_to_request} W"
    else: # それ以上は月単位 (M) や年単位 (Y) を検討するが、ここでは最大を数週間程度に抑えるか、initial_duration_strに近づける
          # あまりに長い期間を毎回計算してリクエストするのは非効率なので、
          # ある程度以上離れていたら initial_duration_str を使うなどの判断もアリ
        # ここでは単純に日数で返す (IBKR APIの上限に注意)
        # 例: 最大でも "4 W" や "1 M" 程度に抑えるなど
        capped_days = min(days_to_request, 30) # 例として最大30日分
        return f"{capped_days} D"


def update_or_create_ohlc_csv(
    base_contract_for_ohlc: Contract,
    contract_month_yyyymm: str,
    host: str,
    port: int,
    client_id: int,
    bar_size_setting: str,
    initial_duration_str: str, # CSVがない場合に使う初期取得期間
    output_csv_filename: str
    ):
    """
    既存のOHLC CSVを更新するか、なければ新規作成する。
    """
    print(f"\n--- Starting CSV Update/Create for: {output_csv_filename} ---")
    print(f"Contract: {contract_month_yyyymm}, Bar: {bar_size_setting}")

    existing_df = None
    last_data_datetime_obj = None # datetimeオブジェクトとして最終日時を保持
    actual_duration_str = initial_duration_str

    if os.path.exists(output_csv_filename) and os.path.getsize(output_csv_filename) > 0:
        try:
            # CSVを読み込む際、Date列を文字列として読み込む
            existing_df = pd.read_csv(output_csv_filename, dtype={'Date': str})
            if not existing_df.empty and "Date" in existing_df.columns:
                # 最終行のDate文字列を取得
                last_date_str_from_csv = existing_df["Date"].iloc[-1]
                last_data_datetime_obj = parse_datetime_flexible(last_date_str_from_csv)

                if last_data_datetime_obj:
                    print(f"Existing CSV found. Last valid date/time entry: {last_data_datetime_obj.strftime('%Y%m%d %H:%M:%S') if last_data_datetime_obj.hour > 0 or last_data_datetime_obj.minute > 0 or last_data_datetime_obj.second > 0 else last_data_datetime_obj.strftime('%Y%m%d')}")
                    actual_duration_str = calculate_duration_since(last_data_datetime_obj.strftime('%Y%m%d %H:%M:%S'), bar_size_setting) #常に詳細な形式で渡す
                    print(f"Calculated duration for update: {actual_duration_str}")
                else:
                    print(f"Could not parse date from last entry: '{last_date_str_from_csv}'. Will fetch initial duration.")
                    existing_df = None # パース失敗時は新規扱い
            else:
                print("Existing CSV is empty or has no 'Date' column. Will fetch initial duration.")
                existing_df = None # 扱いを新規作成と同じにする
        except pd.errors.EmptyDataError:
            print(f"Existing CSV {output_csv_filename} is empty. Will fetch initial duration.")
            existing_df = None
        except Exception as e:
            print(f"Error reading existing CSV {output_csv_filename}: {e}. Will attempt to fetch initial duration.")
            existing_df = None
    else:
        if os.path.exists(output_csv_filename):
             print(f"Existing CSV {output_csv_filename} is empty (0 bytes). Will fetch initial duration.")
        else:
             print(f"CSV {output_csv_filename} not found. Will fetch initial duration.")


    # --- IBKRからデータを取得 ---
    app = IBapi(
        base_contract_details=base_contract_for_ohlc,
        contract_month_to_request=contract_month_yyyymm,
        bar_size_setting=bar_size_setting,
        duration_string=actual_duration_str
    )
    
    print(f"Connecting to {host}:{port} with clientId:{client_id} for historical data (Bar: {bar_size_setting}, Duration: {actual_duration_str})...")
    app.connect(host, port, client_id)

    if not app.isConnected():
        print(f"Failed to connect. Aborting for {output_csv_filename}.")
        return False

    api_thread = Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()

    print(f"Waiting for historical data (Bar: {bar_size_setting}, Duration: {actual_duration_str}) to be processed...")
    
    timeout_duration = 120
    # 期間文字列に応じてタイムアウトを調整するロジック（より洗練させてもよい）
    if "Y" in actual_duration_str.upper(): timeout_duration = 300
    elif "M" in actual_duration_str.upper(): timeout_duration = 240
    elif "W" in actual_duration_str.upper(): timeout_duration = 180
    
    if app.historical_data_end_event.wait(timeout=timeout_duration):
        print(f"Historical data processing finished for {output_csv_filename}.")
    else:
        print(f"Timeout waiting for historicalDataEnd for {output_csv_filename}. Data might be incomplete.")

    new_data_list = app.data
    
    if app.isConnected():
        print(f"Disconnecting from IB for {output_csv_filename}...")
        app.disconnect()
    api_thread.join(timeout=10)

    if not new_data_list or len(new_data_list) <= 1:
        print(f"No new data fetched from IBKR for {output_csv_filename}.")
        if existing_df is not None and not existing_df.empty:
            print(f"Keeping existing CSV as is: {output_csv_filename}")
            # 既存データがあるので成功として扱うか、あるいは新しいデータがないことを示すフラグを返すか検討
            return True # ここでは既存データがあれば成功とする
        return False

    # --- データのマージ ---
    header = new_data_list[0]
    new_df = pd.DataFrame(new_data_list[1:], columns=header)

    if new_df.empty:
        print(f"No new data rows to process after fetching for {output_csv_filename}.")
        if existing_df is not None and not existing_df.empty:
            print(f"Keeping existing CSV as is: {output_csv_filename}")
            return True
        return False

    # Date列をdatetimeオブジェクトに変換 (より堅牢に)
    new_df['Date_dt'] = new_df['Date'].apply(parse_datetime_flexible)
    new_df = new_df.dropna(subset=['Date_dt']) # パースできなかった行は削除

    final_df_list = []
    if existing_df is not None and not existing_df.empty:
        existing_df['Date_dt'] = existing_df['Date'].apply(parse_datetime_flexible)
        existing_df = existing_df.dropna(subset=['Date_dt'])
        
        # 既存データと新規データを結合
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # 数値列の型変換
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    combined_df = combined_df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Date_dt列で重複を削除し、最新のデータを保持
    combined_df = combined_df.drop_duplicates(subset=['Date_dt'], keep='last')
    # Date_dt列でソート
    combined_df = combined_df.sort_values(by='Date_dt')

    # 元の 'Date' 文字列形式を保持しつつ、ソートや重複排除は 'Date_dt' で行う
    # 書き出し時には 'Date_dt' 列は不要なので削除し、元の 'Date' 列を使用
    # ただし、drop_duplicatesで残った行の 'Date' 文字列を使う
    # combined_df は Date_dt で重複排除・ソート済みなので、この時点での 'Date' 列が正しい
    
    if 'Date_dt' in combined_df.columns:
        def reformat_date_from_datetime(dt_obj):
            if pd.NaT is dt_obj or dt_obj is None:
                return None
            # 常に YYYYMMDD HH:MM:SS 形式で出力
            return dt_obj.strftime('%Y%m%d %H:%M:%S')
        combined_df['Date'] = combined_df['Date_dt'].apply(reformat_date_from_datetime)
        combined_df = combined_df.drop(columns=['Date_dt'])


    # --- CSVファイルへの書き込み ---
    try:
        # ヘッダー順序をIBKRの元データに合わせる (Date, Open, High, Low, Close, Volume)
        output_columns = [h for h in header if h in combined_df.columns] # 元のヘッダー順を尊重
        missing_cols = set(header) - set(output_columns)
        if missing_cols:
            print(f"Warning: Columns {missing_cols} are missing from the final DataFrame. They will not be in the CSV.")

        combined_df.to_csv(output_csv_filename, columns=output_columns, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Successfully updated/created CSV: {output_csv_filename}")
        return True
    except Exception as e:
        print(f"Error writing combined data to CSV {output_csv_filename}: {e}")
        return False
