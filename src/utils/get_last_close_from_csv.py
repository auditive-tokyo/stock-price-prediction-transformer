import os
import pandas as pd

def get_last_close_from_csv(csv_path):
    """OHLC CSVファイルから最新の終値(Close)を読み取る"""
    if not os.path.exists(csv_path):
        return None
    try:
        # ファイルの最終行のみを効率的に読み込む
        df = pd.read_csv(csv_path, usecols=['Close']).tail(1)
        if not df.empty:
            return df['Close'].iloc[0]
    except Exception as e:
        print(f"Could not read last close from {csv_path}: {e}")
    return None