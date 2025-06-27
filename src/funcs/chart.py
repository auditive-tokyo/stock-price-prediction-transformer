import os

def create_charts_from_csvs(
    OHLC_CONFIGS,
    N225_FUTURES_CONTRACT,
    active_contract_month,
    CSV_OUTPUT_DIR,
    CHART_OUTPUT_DIR,
    create_chart_func
):
    generated_chart_paths = []
    generated_chart_timeframes = []
    for bar_size, initial_duration, suffix, chart_title_suffix in OHLC_CONFIGS:
        csv_filename_only = f"{N225_FUTURES_CONTRACT.symbol}_ohlc_{active_contract_month}_{suffix}.csv"
        csv_output_file = os.path.join(CSV_OUTPUT_DIR, csv_filename_only)

        if os.path.exists(csv_output_file):
            print(f"\nCreating chart for {csv_output_file}...")
            chart_filename_only = f"{N225_FUTURES_CONTRACT.symbol}_chart_{active_contract_month}_{suffix}.jpg"
            chart_output_file = os.path.join(CHART_OUTPUT_DIR, chart_filename_only)
            chart_title = f"日経225先物 {active_contract_month} ({chart_title_suffix})"

            try:
                chart_image_path = create_chart_func(
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

    print("\n--- Creating Charts Done ---")
    return generated_chart_paths, generated_chart_timeframes