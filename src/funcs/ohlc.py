import os
import time

def update_ohlc_csvs_for_configs(
    OHLC_CONFIGS,
    N225_FUTURES_CONTRACT,
    active_contract_month,
    HOST,
    PORT,
    CLIENT_ID_OHLC_DATA_BASE,
    CSV_OUTPUT_DIR,
    update_or_create_ohlc_csv_func
):
    ohlc_client_id_counter = 0
    for bar_size, initial_duration, suffix, chart_title_suffix in OHLC_CONFIGS:
        current_ohlc_client_id = CLIENT_ID_OHLC_DATA_BASE + ohlc_client_id_counter

        csv_filename_only = f"{N225_FUTURES_CONTRACT.symbol}_ohlc_{active_contract_month}_{suffix}.csv"
        csv_output_file = os.path.join(CSV_OUTPUT_DIR, csv_filename_only)

        print(f"\nProcessing CSV for contract {active_contract_month}, Bar: {bar_size} (Initial Duration: {initial_duration}) (clientId: {current_ohlc_client_id})...")
        print(f"Target CSV file: {csv_output_file}")

        csv_updated_successfully = update_or_create_ohlc_csv_func(
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