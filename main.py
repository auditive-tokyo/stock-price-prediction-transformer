import datetime
import time
import os
from ibapi.contract import Contract
from src.utils.ib_futures_expirations import get_n225_futures_expirations
from src.utils.active_contract_month import determine_active_contract_month
from src.funcs.update_ohlc_data import update_or_create_ohlc_csv
from src.funcs.create_rosoku import create_chart # create_chart をインポート

# --- Global Settings ---
HOST = "127.0.0.1"
PORT = 7497

N225_FUTURES_CONTRACT = Contract()
N225_FUTURES_CONTRACT.symbol = "N225"
N225_FUTURES_CONTRACT.secType = "FUT"
N225_FUTURES_CONTRACT.exchange = "OSE.JPN"
N225_FUTURES_CONTRACT.currency = "JPY"

CLIENT_ID_EXPIRATIONS = 55
CLIENT_ID_OHLC_DATA_BASE = 56

# CSV保存先ディレクトリ
CSV_OUTPUT_DIR = "chart_csvs"
# チャート画像保存先ディレクトリ
CHART_OUTPUT_DIR = "chart_images" # 追加

# 取得する足種と期間の組み合わせ
# (barSizeSetting, initial_durationStr_for_new_csv, file_suffix, chart_title_suffix)
OHLC_CONFIGS = [
    ("15 mins", "1 Y", "15M", "15分足"), # チャートタイトル用のサフィックスを追加
    ("4 hours", "5 Y", "4H", "4時間足"),   # チャートタイトル用のサフィックスを追加
    # ("1 day", "5 Y", "1D", "日足")
]
# --- End Global Settings ---

def main():
    print("--- Main Script Started ---")

    # CSV保存先ディレクトリを作成 (存在しない場合)
    if not os.path.exists(CSV_OUTPUT_DIR):
        os.makedirs(CSV_OUTPUT_DIR)
        print(f"Created directory: {CSV_OUTPUT_DIR}")

    # チャート画像保存先ディレクトリを作成 (存在しない場合)
    if not os.path.exists(CHART_OUTPUT_DIR): # 追加
        os.makedirs(CHART_OUTPUT_DIR)       # 追加
        print(f"Created directory: {CHART_OUTPUT_DIR}") # 追加


    # --- 1. Fetch N225 Futures Expirations ---
    print(f"\nStep 1: Calling get_n225_futures_expirations (clientId: {CLIENT_ID_EXPIRATIONS})...")
    n225_expirations_yyyymmdd = get_n225_futures_expirations(
        base_contract=N225_FUTURES_CONTRACT,
        host=HOST,
        port=PORT,
        client_id=CLIENT_ID_EXPIRATIONS
    )

    if not n225_expirations_yyyymmdd:
        print("Failed to retrieve N225 futures expiration dates. Exiting.")
        return

    # --- 2. Determine Current Date and Select Contract Month ---
    current_date = datetime.date.today()
    print(f"\nStep 2: Selecting contract month based on current date: {current_date}")
    active_contract_month = determine_active_contract_month(n225_expirations_yyyymmdd, current_date)

    if not active_contract_month:
        print("\nStep 3: Could not determine an active N225 contract month. Exiting.")
        return
    
    print(f"\nStep 3: The active N225 contract month to be used is: {active_contract_month}")


    # --- 4. Update or Create OHLC CSVs for each configuration ---
    print("\n--- Step 4: Updating/Creating OHLC CSVs ---")
    ohlc_client_id_counter = 0
    for bar_size, initial_duration, suffix, chart_title_suffix in OHLC_CONFIGS:
        current_ohlc_client_id = CLIENT_ID_OHLC_DATA_BASE + ohlc_client_id_counter
        
        csv_filename_only = f"n225_ohlc_{active_contract_month}_{suffix}.csv"
        csv_output_file = os.path.join(CSV_OUTPUT_DIR, csv_filename_only)

        print(f"\nProcessing CSV for contract {active_contract_month}, Bar: {bar_size} (Initial Duration: {initial_duration}) (clientId: {current_ohlc_client_id})...")
        print(f"Target CSV file: {csv_output_file}")
        
        csv_updated_successfully = update_or_create_ohlc_csv(
            base_contract_for_ohlc=N225_FUTURES_CONTRACT,
            contract_month_yyyymm=active_contract_month,
            host=HOST,
            port=PORT,
            client_id=current_ohlc_client_id,
            bar_size_setting=bar_size,
            initial_duration_str=initial_duration,
            output_csv_filename=csv_output_file
        )

        if csv_updated_successfully:
            print(f"Successfully updated/created CSV: {csv_output_file}")
        else:
            print(f"Failed to update/create CSV: {csv_output_file}")
        
        ohlc_client_id_counter += 1
        print("Waiting for 5 seconds before next IB API operation...")
        time.sleep(5)
    
    # --- 5. Create Charts from CSVs for each configuration ---
    print("\n--- Step 5: Creating Charts from CSVs ---")
    for bar_size, initial_duration, suffix, chart_title_suffix in OHLC_CONFIGS:
        csv_filename_only = f"n225_ohlc_{active_contract_month}_{suffix}.csv"
        csv_output_file = os.path.join(CSV_OUTPUT_DIR, csv_filename_only)

        if os.path.exists(csv_output_file):
            print(f"\nCreating chart for {csv_output_file}...")
            chart_filename_only = f"n225_chart_{active_contract_month}_{suffix}.jpg"
            chart_output_file = os.path.join(CHART_OUTPUT_DIR, chart_filename_only)
            chart_title = f"日経225先物 {active_contract_month} ({chart_title_suffix})"

            try:
                chart_image_path = create_chart(
                    csv_file=csv_output_file,
                    output_file=chart_output_file,
                    title=chart_title
                )
                print(f"Successfully created chart: {chart_image_path}")
            except Exception as e:
                print(f"Error creating chart for {csv_output_file}: {e}")
        else:
            print(f"Skipping chart creation for {csv_output_file} as it does not exist.")

    print("\n--- Main Script Finished ---")

if __name__ == "__main__":
    main()