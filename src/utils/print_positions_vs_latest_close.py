import os

def print_positions_vs_latest_close(
    current_positions,
    N225_FUTURES_CONTRACT,
    active_contract_month,
    OHLC_CONFIGS,
    CSV_OUTPUT_DIR,
    get_last_close_from_csv
):
    """
    現在のポジションと最新CSV終値を比較して出力する
    """
    if not OHLC_CONFIGS:
        print("OHLC_CONFIGS is empty.")
        return

    first_config_suffix = OHLC_CONFIGS[0][2]  # 例: "15M"
    csv_filename = f"{N225_FUTURES_CONTRACT.symbol}_ohlc_{active_contract_month}_{first_config_suffix}.csv"
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
        if (
            latest_close_price is not None
            and pos['symbol'] == N225_FUTURES_CONTRACT.symbol
            and pos['lastTradeDateOrContractMonth'].startswith(active_contract_month[:6])
        ):
            print(f"    Latest CSV Close: {latest_close_price:.2f}")
            # 損益計算では元の符号付き数量を使用する
            pnl_per_unit = (latest_close_price - avg_price) * position_qty
            print(f"    Unrealized P/L (vs latest close): {pnl_per_unit:.2f}")

        print("-" * 20)

    return latest_close_price