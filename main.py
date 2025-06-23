import datetime
import time
import os
from ibapi.contract import Contract
from src.utils.ib_futures_expirations import get_n225_futures_expirations
from src.utils.active_contract_month import determine_active_contract_month
from src.utils.get_last_close_from_csv import get_last_close_from_csv
from src.funcs.get_positions import get_current_positions
from src.funcs.update_ohlc_data import update_or_create_ohlc_csv
from src.funcs.create_rosoku import create_chart
from src.funcs.get_and_summarize_top_articles import get_and_summarize_top_articles

# --- Global Settings ---
HOST = "127.0.0.1"
# PORT = 7497
PORT = 4001 # 本番

N225_FUTURES_CONTRACT = Contract()
N225_FUTURES_CONTRACT.symbol = "N225"
N225_FUTURES_CONTRACT.secType = "FUT"
N225_FUTURES_CONTRACT.exchange = "OSE.JPN"
N225_FUTURES_CONTRACT.currency = "JPY"

CLIENT_ID_EXPIRATIONS = 55
CLIENT_ID_OHLC_DATA_BASE = 56
CLIENT_ID_POSITIONS = 99

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
        print("\nStep 2: Could not determine an active N225 contract month. Exiting.")
        return
    
    print(f"\nStep 2: The active N225 contract month to be used is: {active_contract_month}")


    # --- 3. Update or Create OHLC CSVs for each configuration ---
    print("\n--- Step 3: Updating/Creating OHLC CSVs ---")
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


    # --- 4. Create Charts from CSVs for each configuration ---
    print("\n--- Step 4: Creating Charts from CSVs ---")
    generated_chart_paths = []
    generated_chart_timeframes = []
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
                generated_chart_paths.append(chart_image_path)
                generated_chart_timeframes.append(chart_title_suffix)
            except Exception as e:
                print(f"Error creating chart for {csv_output_file}: {e}")
        else:
            print(f"Skipping chart creation for {csv_output_file} as it does not exist.")

    print("\n--- Main Script Finished ---")


    # --- 5. Get and Format Summaries of Top News Articles ---
    print("\n--- Step 5: Getting and Formatting Summaries of Top News Articles ---")
    
    # 5-1. 関数を呼び出し、要約文のリストを取得
    summaries_list = get_and_summarize_top_articles(top_n=5)

    # 5-2. AIに渡すために、取得したリストを整形済みの単一文字列に変換
    formatted_summaries_for_ai = ""
    if summaries_list:
        # 番号付きリストの各行を作成
        summary_lines = [f"{i}. {summary}" for i, summary in enumerate(summaries_list, 1)]
        
        # ヘッダーと改行コードを使って最終的な文字列を組み立てる
        formatted_summaries_for_ai = "Top 5 News Summaries:\n" + "\n\n".join(summary_lines)
        
        print("\n--- Formatted Summaries (for AI Prompt) ---")
        print(formatted_summaries_for_ai)
        print("---------------------------------------------")
    else:
        print("\nCould not retrieve any article summaries.")


    # --- 6. Display Current Positions vs Latest CSV Price ---
    print("\n--- Step 6: Getting Current Positions and Comparing with Latest CSV Price ---")
    current_positions = get_current_positions(
        host=HOST,
        port=PORT,
        clientId=CLIENT_ID_POSITIONS
    )

    if current_positions is None:
        print("Could not retrieve positions due to an error.")
    elif not current_positions:
        print("No open positions found.")
    else:
        print("\n--- Current Open Positions vs Latest Close Price ---")
        # 比較対象のCSVファイルとして、最も粒度の細かい(最初の)設定を使用
        if OHLC_CONFIGS:
            first_config_suffix = OHLC_CONFIGS[0][2] # 例: "15M"
            csv_filename = f"n225_ohlc_{active_contract_month}_{first_config_suffix}.csv"
            csv_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)
            
            latest_close_price = get_last_close_from_csv(csv_path)
            
            if latest_close_price is not None:
                print(f"(Using latest close price from: {csv_path})")
            else:
                print(f"(Warning: Could not read latest close price from {csv_path})")

            for pos in current_positions:
                position_qty = pos.get('position', 0)
                avg_price = pos.get('avgPrice', 0)
                
                # ポジションの方向（買い/売り）を判定
                side = "Long" if position_qty > 0 else "Short" if position_qty < 0 else "Flat"

                print(f"\n  Account: {pos['account']}")
                print(f"    Symbol: {pos['symbol']}, Expiry: {pos['lastTradeDateOrContractMonth']}")
                # 数量は絶対値を表示し、Side（方向）を明記する
                print(f"    Side: {side}, Quantity: {abs(position_qty)}, Avg Price: {avg_price:.2f}")
                
                # ポジションが現在処理中の限月と一致する場合のみ価格比較を行う
                if latest_close_price is not None and \
                   pos['symbol'] == N225_FUTURES_CONTRACT.symbol and \
                   pos['lastTradeDateOrContractMonth'].startswith(active_contract_month[:6]):
                    
                    print(f"    Latest CSV Close: {latest_close_price:.2f}")
                    # 損益計算では元の符号付き数量を使用する
                    pnl_per_unit = (latest_close_price - avg_price) * position_qty
                    print(f"    Unrealized P/L (vs latest close): {pnl_per_unit:.2f}")
                
                print("-" * 20)


    # # --- 7. Analyze Charts with OpenAI ---
    # print("\n--- Step 7: Analyzing charts with OpenAI ---")
    # analysis_result = None
    # if generated_chart_paths:
    #     try:
    #         print("Sending charts to OpenAI for analysis...")
    #         analysis_result = analyze_chart_with_function_calling(
    #             image_paths=generated_chart_paths,
    #             timeframes=generated_chart_timeframes
    #         )
    #         print("\n--- OpenAI Analysis Result ---")
    #         print(f"  Decision: {analysis_result.get('decision', 'N/A')}")
    #         print(f"  Confidence: {analysis_result.get('confidence', 'N/A')}")
    #         print(f"  Reason: {analysis_result.get('reason', 'N/A')}")
    #         if analysis_result.get('additional_info_needed'):
    #             print(f"  Additional Info Needed: {', '.join(analysis_result['additional_info_needed'])}")
    #         print("------------------------------")

    #     except Exception as e:
    #         print(f"An error occurred during OpenAI analysis: {e}")
    # else:
    #     print("No charts were generated, skipping OpenAI analysis.")

if __name__ == "__main__":
    main()