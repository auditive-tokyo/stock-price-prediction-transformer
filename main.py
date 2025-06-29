import datetime
import time
import os
from ibapi.contract import Contract
from src.utils.ib_futures_expirations import get_n225_futures_expirations
from src.utils.active_contract_month import determine_active_contract_month
from src.utils.get_last_close_from_csv import get_last_close_from_csv
from src.utils.print_positions_vs_latest_close import print_positions_vs_latest_close
from src.utils.save_analysis_to_csv import save_analysis_to_csv
from src.funcs.get_positions import get_current_positions
from src.funcs.update_ohlc_data import update_or_create_ohlc_csv
from src.funcs.create_rosoku import create_chart
from src.funcs.get_and_summarize_top_articles import get_and_summarize_top_articles
from src.funcs.openai_prediction import analyze_chart_with_function_calling
from src.funcs.openai_transaction import analyze_close_decision_with_function_calling
from src.funcs.show_margin import get_account_updates
from src.funcs.get_board_info import get_board_info_markdown
from src.funcs.chart import create_charts_from_csvs
from src.funcs.close_position import close_n225_position
from src.funcs.place_order import place_n225_market_order
from src.funcs.ohlc import update_ohlc_csvs_for_configs

# --- Global Settings ---
HOST = "127.0.0.1"
PORT = 7497
# PORT = 4001 # 本番

N225_FUTURES_CONTRACT = Contract()
N225_FUTURES_CONTRACT.symbol = "N225"
N225_FUTURES_CONTRACT.secType = "FUT"
N225_FUTURES_CONTRACT.exchange = "OSE.JPN"
N225_FUTURES_CONTRACT.currency = "JPY"

MAX_ORDER_NUMBER = 2

# クライアントIDの設定（特に変更は不要）
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
    ("1 day", "5 Y", "1D", "日足")
]
# --- End Global Settings ---

def main():
    while True:
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
        print(f"\nStep 1: Calling get_{N225_FUTURES_CONTRACT.symbol}_futures_expirations (clientId: {CLIENT_ID_EXPIRATIONS})...")
        n225_expirations_yyyymmdd = get_n225_futures_expirations(
            base_contract=N225_FUTURES_CONTRACT,
            host=HOST,
            port=PORT,
            client_id=CLIENT_ID_EXPIRATIONS
        )

        if not n225_expirations_yyyymmdd:
            print(f"Failed to retrieve {N225_FUTURES_CONTRACT.symbol} futures expiration dates. Exiting.")
            return


        # --- 2. Determine Current Date and Select Contract Month ---
        current_date = datetime.date.today()
        print(f"\nStep 2: Selecting contract month based on current date: {current_date}")
        active_contract_month = determine_active_contract_month(n225_expirations_yyyymmdd, current_date)

        if not active_contract_month:
            print(f"\nStep 2: Could not determine an active {N225_FUTURES_CONTRACT.symbol} contract month. Exiting.")
            return
        
        print(f"\nStep 2: The active {N225_FUTURES_CONTRACT.symbol} contract month to be used is: {active_contract_month}")


        # --- 3. Update or Create OHLC CSVs for each configuration ---
        print("\n--- Step 3: Updating/Creating OHLC CSVs ---")
        update_ohlc_csvs_for_configs(
            OHLC_CONFIGS=OHLC_CONFIGS,
            N225_FUTURES_CONTRACT=N225_FUTURES_CONTRACT,
            active_contract_month=active_contract_month,
            HOST=HOST,
            PORT=PORT,
            CLIENT_ID_OHLC_DATA_BASE=CLIENT_ID_OHLC_DATA_BASE,
            CSV_OUTPUT_DIR=CSV_OUTPUT_DIR,
            update_or_create_ohlc_csv_func=update_or_create_ohlc_csv
        )


        # --- 4. Create Charts from CSVs for each configuration ---
        generated_chart_paths, generated_chart_timeframes = create_charts_from_csvs(
            OHLC_CONFIGS=OHLC_CONFIGS,
            N225_FUTURES_CONTRACT=N225_FUTURES_CONTRACT,
            active_contract_month=active_contract_month,
            CSV_OUTPUT_DIR=CSV_OUTPUT_DIR,
            CHART_OUTPUT_DIR=CHART_OUTPUT_DIR,
            create_chart_func=create_chart
        )


        # --- 5. Get and Format Summaries of Top News Articles ---
        print("\n--- Step 5: Getting and Formatting Summaries of Top News Articles ---")
        
        # 5-1. 関数を呼び出し、要約文のリストを取得
        news_summaries = get_and_summarize_top_articles(top_n=5)

        # 5-2. AIに渡すために、取得したリストを整形済みの単一文字列に変換
        formatted_summaries_for_ai = ""
        if news_summaries:
            # 番号付きリストの各行を作成
            summary_lines = [f"{i}. {summary}" for i, summary in enumerate(news_summaries, 1)]
            
            # ヘッダーと改行コードを使って最終的な文字列を組み立てる
            formatted_summaries_for_ai = "Top 5 News Summaries:\n" + "\n\n".join(summary_lines)
            
            print("\n--- Formatted Summaries (for AI Prompt) ---")
            print(formatted_summaries_for_ai)
            print("---------------------------------------------")
        else:
            print("\nCould not retrieve any article summaries.")


        # --- 6. Get the Board Inforamation ---
        board_info_md = get_board_info_markdown()
        print(f"Board Info: \n{board_info_md}")


        # --- 7. Display Current Positions vs Latest CSV Price ---
        print("\n--- Step 7: Getting Current Positions and Comparing with Latest CSV Price ---")
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
            latest_close_price = print_positions_vs_latest_close(
                current_positions=current_positions,
                N225_FUTURES_CONTRACT=N225_FUTURES_CONTRACT,
                active_contract_month=active_contract_month,
                OHLC_CONFIGS=OHLC_CONFIGS,
                CSV_OUTPUT_DIR=CSV_OUTPUT_DIR,
                get_last_close_from_csv=get_last_close_from_csv
            )

            # 決済判断ロジックを追加
            print("\n--- Step X: Deciding whether to close positions ---")
            # 例: 各ポジションの損益や条件を見て決済判断
            for pos in current_positions:
                transaction_decision = analyze_close_decision_with_function_calling(
                    image_paths=generated_chart_paths,
                    timeframes=generated_chart_timeframes, 
                    current_position=pos,  # 1つだけ渡す
                    news_summaries=formatted_summaries_for_ai,
                    board_info_md=board_info_md,
                    latest_close_price=latest_close_price
                )
                print(f"\n--- OpenAI Close Decision Result for {pos['symbol']} ({pos['lastTradeDateOrContractMonth']}) ---")
                print(f"  Decision: {transaction_decision.get('decision', 'N/A')}")
                print(f"  Confidence: {transaction_decision.get('confidence', 'N/A')}")
                print(f"  Reason: {transaction_decision.get('reason', 'N/A')}")
                if transaction_decision.get('additional_info_needed'):
                    print(f"  Additional Info Needed: {', '.join(transaction_decision['additional_info_needed'])}")
                print("------------------------------")

                # 決済判断
                if transaction_decision.get('decision') == "決済":
                    print("→ ポジションを決済します")
                    result = close_n225_position(
                        current_position=pos,
                        expiration_month=pos["lastTradeDateOrContractMonth"][:6]
                    )
                    print(f"  Close order result: {result}")
                else:
                    print("→ 継続保有（何もしません）")

                save_analysis_to_csv(transaction_decision, situation="決済")

        # --- 7.5: Getting Margin Info (AvailableFunds)  ---
        print("\n--- Step 7.5: Getting Margin Info (AvailableFunds) ---")
        available_funds = get_account_updates(host=HOST, port=PORT, clientId=1001)
        print(f"AvailableFunds (JPY): {available_funds}")

        # TODO: 1枚あたりの必要証拠金を計算（あってるか知らない）
        latest_price = latest_close_price  # 直近の終値
        if N225_FUTURES_CONTRACT.symbol == "N225":
            contract_multiplier = 1000  # 日経225先物の取引単位
        elif N225_FUTURES_CONTRACT.symbol == "N225M":
            contract_multiplier = 100  # 日経225先物ミニの取引単位

        order_size = 1                     # 新規注文枚数
        margin_rate = 0.10                 # 証拠金率（10%の場合）

        required_margin = latest_price * contract_multiplier * order_size * margin_rate
        print(f"必要証拠金（概算）: {required_margin:.0f} 円")

        # フラグ初期化
        can_place_order = False
        insufficient_margin = False
        funds_unavailable = False

        if available_funds is not None:
            if available_funds > required_margin:
                print("新規注文可能です。")
                can_place_order = True
            else:
                print("証拠金不足です。")
                insufficient_margin = True
        else:
            print("AvailableFundsが取得できませんでした。")
            funds_unavailable = True


        # --- 8. Decide weather to buy, sell, or wait ---
        print("\n--- Step 8: Deciding whether to buy, sell, or wait ---")

        # 現在の保有枚数を全銘柄合計でカウント
        total_position_qty = 0
        if current_positions:
            for pos in current_positions:
                total_position_qty += abs(pos.get('position', 0))

        if can_place_order:
            if total_position_qty >= MAX_ORDER_NUMBER:
                print(f"Order NOT placed: 現在の保有枚数({total_position_qty})が最大枚数({MAX_ORDER_NUMBER})に達しています。")
                print("最大保有枚数制限のため、新規注文はスキップされます。")
            else:
                print("Proceeding with order placement...")
                # --- 9. Analyze Charts with OpenAI ---
                print("\n--- Step 9: Analyzing charts with OpenAI ---")
                analysis_result = None
                try:
                    print("Sending charts to OpenAI for analysis...")
                    analysis_result = analyze_chart_with_function_calling(
                        image_paths=generated_chart_paths,
                        timeframes=generated_chart_timeframes,
                        news_summaries=formatted_summaries_for_ai,
                        board_info_md=board_info_md,
                        latest_close_price=latest_close_price
                    )
                    print("\n--- OpenAI Analysis Result ---")
                    print(f"  Decision: {analysis_result.get('decision', 'N/A')}")
                    print(f"  Confidence: {analysis_result.get('confidence', 'N/A')}")
                    print(f"  Reason: {analysis_result.get('reason', 'N/A')}")
                    if analysis_result.get('additional_info_needed'):
                        print(f"  Additional Info Needed: {', '.join(analysis_result['additional_info_needed'])}")
                    print("------------------------------")

                    match analysis_result.get('decision'):
                        case "買い":
                            print("→ 成り行き買い注文を実行")
                            order_result = place_n225_market_order(
                                action="BUY",
                                quantity=order_size,  # 1で固定している
                                expiration_month=active_contract_month,
                                host=HOST,
                                port=PORT,
                                clientId=101  # 他と重複しないIDを指定（例: 101）
                            )
                            print(f"  Order result: {order_result}")
                        case "売り":
                            print("→ 成り行き売り注文を実行")
                            order_result = place_n225_market_order(
                                action="SELL",
                                quantity=order_size,
                                expiration_month=active_contract_month,
                                host=HOST,
                                port=PORT,
                                clientId=102  # 他と重複しないIDを指定（例: 102）
                            )
                            print(f"  Order result: {order_result}")
                        case "待ち":
                            print("→ 今回は見送り（待ち）")
                        case _:
                            print("→ 分析エラーまたは判定不能")

                    save_analysis_to_csv(analysis_result, situation="オーダー")

                except Exception as e:
                    print(f"An error occurred during OpenAI analysis: {e}")
        elif insufficient_margin:
            print("Insufficient margin to place an order. Skipping order placement.")
        elif funds_unavailable:
            print("Funds information is unavailable. Cannot decide on order placement.")




        # 15分待機して次のループへ
        print("\n--- Waiting 15 minutes before next execution ---")
        time.sleep(15 * 60)

if __name__ == "__main__":
    main()