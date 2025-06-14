import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import japanize_matplotlib
import os

# RSIを計算する関数
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# ボリンジャーバンドを計算する関数
def calculate_bollinger_bands(data, window=20, num_std=2):
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

# MACDを計算する関数
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    # 短期EMA
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    # 長期EMA
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    # MACD線 = 短期EMA - 長期EMA
    macd_line = fast_ema - slow_ema
    # シグナル線 = MACDのEMA
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    # ヒストグラム = MACD線 - シグナル線
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def create_chart(csv_file, output_file=None, title='日経平均先物価格推移', dpi=300):
    """
    CSVデータからローソク足チャートを作成し、ファイルに保存する
    
    Parameters:
    -----------
    csv_file : str
        ローソク足データを含むCSVファイルのパス (UTF-8エンコード、ヘッダー: Date,Open,High,Low,Close,Volume を想定)
    output_file : str, optional
        出力画像ファイルのパス（デフォルトはNone。指定がない場合はCSVファイル名から自動生成）
    title : str, optional
        チャートのタイトル（デフォルトは'日経平均先物価格推移'）
    dpi : int, optional
        出力画像の解像度（デフォルトは300）
        
    Returns:
    --------
    str : 保存された画像ファイルのパス
    """
    
    # 出力ファイル名が指定されていない場合はCSVファイル名から自動生成
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = f"{base_name}_chart.jpg"
    
    # CSVファイルをUTF-8エンコードで読み込む
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return None
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return None

    # 必要なカラムが存在するか確認
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file {csv_file} is missing one or more required columns: {required_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Date列を日時型に変換
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Error converting 'Date' column to datetime in {csv_file}: {e}")
        return None

    # Date列をインデックスに設定
    df = df.set_index('Date')

    # 念のため、数値列を数値型に変換 (update_ohlc_data.pyでも行っているが、ここでも行うとより堅牢)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])


    # 最新x00行だけに絞る (チャートの見やすさのため)
    if len(df) > 200:
        df = df.tail(100)
    elif df.empty:
        print(f"Warning: DataFrame is empty after processing {csv_file}. Cannot create chart.")
        return None

    # 移動平均線の計算 (5期間と25期間)
    # 注意: 15分足と4時間足で同じ期間のMAで良いかは検討の余地あり
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA25'] = df['Close'].rolling(window=25).mean()

    # RSIの計算 (14期間)
    df['RSI'] = calculate_rsi(df['Close'], period=14)

    # ボリンジャーバンドの計算（20期間、標準偏差2）
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], window=20, num_std=2)

    # MACDの計算（短期12、長期26、シグナル9）
    df['MACD_Line'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'], fast_period=12, slow_period=26, signal_period=9)

    # 色の定義
    ma5_color = '#FF8C00'  # オレンジ色
    ma25_color = '#4B0082'  # インディゴ色
    rsi_color = '#008000'   # 緑色
    bb_color = '#1E90FF'    # ドジャーブルー
    macd_line_color = '#FF4500'  # オレンジレッド
    macd_signal_color = '#00CED1'  # ダークターコイズ
    macd_histogram_positive_color = '#00FF00'  # ライム
    macd_histogram_negative_color = '#FF6347'  # トマト

    # カスタムスタイルの作成
    mc = mpf.make_marketcolors(
        up='red',
        down='blue',
        edge='inherit',
        wick='inherit',
        volume='yellow',  # 出来高を黄色に設定
    )

    # japanize_matplotlibのフォント設定をmplfinanceに渡す
    s = mpf.make_mpf_style(
        marketcolors=mc,
        rc={'font.family': plt.rcParams['font.family']},
        gridstyle='--',  # 破線のグリッド
        gridcolor='gray',  # グリッドの色
        gridaxis='both'    # 縦横両方にグリッドを表示
    )

    # ヒストグラムを正と負に分ける
    positive_histogram = df['MACD_Histogram'].where(df['MACD_Histogram'] >= 0, 0)
    negative_histogram = df['MACD_Histogram'].where(df['MACD_Histogram'] < 0, 0)

    # 追加プロット（移動平均線、ボリンジャーバンド、RSI、MACD）の設定
    additional_plots = [
        mpf.make_addplot(df['MA5'], color=ma5_color, width=1.5),  # オレンジ色
        mpf.make_addplot(df['MA25'], color=ma25_color, width=1.5),  # インディゴ色
        mpf.make_addplot(df['BB_Upper'], color=bb_color, width=1, linestyle='--'),  # 上のバンド
        mpf.make_addplot(df['BB_Middle'], color=bb_color, width=1),  # 中央のバンド
        mpf.make_addplot(df['BB_Lower'], color=bb_color, width=1, linestyle='--'),  # 下のバンド
        mpf.make_addplot(df['RSI'], panel=1, color=rsi_color, width=1.5, ylabel='RSI'),  # RSIを別パネルに表示
        mpf.make_addplot(df['MACD_Line'], panel=2, color=macd_line_color, width=1.5, ylabel='MACD', secondary_y=False),  # MACD線
        mpf.make_addplot(df['MACD_Signal'], panel=2, color=macd_signal_color, width=1.5, secondary_y=False),  # シグナル線
        mpf.make_addplot(positive_histogram, panel=2, color=macd_histogram_positive_color, type='bar', width=0.7, secondary_y=False),  # 正のヒストグラム
        mpf.make_addplot(negative_histogram, panel=2, color=macd_histogram_negative_color, type='bar', width=0.7, secondary_y=False)  # 負のヒストグラム
    ]

    # MACDのゼロライン
    macd_zero = pd.Series(0, index=df.index)
    additional_plots.append(mpf.make_addplot(macd_zero, panel=2, color='gray', linestyle='--', width=0.8))

    # ローソク足チャートを描画、凡例を追加
    fig, axes = mpf.plot(df, 
                        type='candle', 
                        title=title,
                        ylabel='価格',
                        volume=True, 
                        volume_panel=3,  # 出来高パネルは3番目
                        figsize=(12, 12),  # 高さを調整
                        style=s,
                        addplot=additional_plots,
                        panel_ratios=(6, 2, 2, 2),  # 価格:RSI:MACD:出来高 の比率
                        main_panel=0,  
                        returnfig=True)  # 図とaxesを返す

    # 凡例用のハンドルを作成
    ma5_line = mlines.Line2D([], [], color=ma5_color, linewidth=1.5, label='MA5')
    ma25_line = mlines.Line2D([], [], color=ma25_color, linewidth=1.5, label='MA25')
    bb_line = mlines.Line2D([], [], color=bb_color, linewidth=1, label='BB(20,2)')
    rsi_line = mlines.Line2D([], [], color=rsi_color, linewidth=1.5, label='RSI(14)')
    macd_line = mlines.Line2D([], [], color=macd_line_color, linewidth=1.5, label='MACD')
    macd_signal = mlines.Line2D([], [], color=macd_signal_color, linewidth=1.5, label='Signal')

    # 凡例を追加 - 正しいパネルに
    axes[0].legend(handles=[ma5_line, ma25_line, bb_line], loc='upper left')
    axes[2].legend(handles=[rsi_line], loc='upper left')  # RSIパネル
    axes[4].legend(handles=[macd_line, macd_signal], loc='upper left')  # MACDパネル

    rsi_ax = axes[2]
    rsi_ax.set_ylim(0, 100)  # 強制的に0-100に設定
    rsi_ax.set_yticks([25, 50, 75])  # 目盛りを設定

    # MACDパネルを0を中心に対称的に表示する
    macd_ax = axes[4]
    # MACDの最大絶対値を取得して、その値を使って対称的なY軸範囲を設定
    macd_max = max(
        abs(df['MACD_Line'].max()),
        abs(df['MACD_Line'].min()),
        abs(df['MACD_Signal'].max()),
        abs(df['MACD_Signal'].min()),
        abs(df['MACD_Histogram'].max()),
        abs(df['MACD_Histogram'].min())
    )
    # 余裕を持たせるため20%増しにする
    macd_limit = macd_max * 1.2
    # Y軸の範囲を設定（0を中心に対称的に）
    macd_ax.set_ylim(-macd_limit, macd_limit)
    # ゼロラインを強調表示
    macd_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # 保存
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)  # メモリリークを防ぐためにfigをクローズ
    
    return output_file


# --- ここから単体テスト用のコード ---
if __name__ == '__main__':
    print("--- create_rosoku.py direct execution test ---")

    # --- テスト用の設定 ---
    # 実際のCSVファイルが格納されているディレクトリ
    # このスクリプト (create_rosoku.py) から見た相対パス、または絶対パスを指定
    # create_rosoku.py は src/funcs/ にあるので、プロジェクトルートからの相対パスは ../../chart_csvs
    ACTUAL_CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chart_csvs")
    
    TEST_IMAGE_DIR = "test_chart_images_from_actual_data" # テスト用画像を保存するディレクトリ
    
    if not os.path.exists(ACTUAL_CSV_DIR):
        print(f"ERROR: CSV directory not found at {ACTUAL_CSV_DIR}")
        print("Please ensure that 'chart_csvs' directory exists and contains OHLC CSV files.")
        exit()

    if not os.path.exists(TEST_IMAGE_DIR):
        os.makedirs(TEST_IMAGE_DIR)
        print(f"Created directory for test images: {TEST_IMAGE_DIR}")

    # --- テスト対象のCSVファイルと設定 ---
    # chart_csvs ディレクトリ内のCSVファイルを自動で検出するか、手動で指定する
    # ここでは、特定の命名規則のファイルを対象とする例
    
    # 例: 'n225_ohlc_YYYYMM_suffix.csv' のようなファイル名パターンを想定
    # YYYYMM は実行時の状況によって変わるため、ここでは固定値または動的取得が必要
    # 簡単のため、よく使われるであろうファイル名を直接指定する例を示します。
    # 必要に応じて、glob などでファイル一覧を取得するように変更してください。
    
    # --- !!! 注意: 以下のファイル名と限月は、ご自身の環境に合わせてください !!! ---
    active_contract_month_for_test = "202509" # テストしたい契約月

    test_csv_files_info = [
        {
            "filename": f"n225_ohlc_{active_contract_month_for_test}_15min.csv", 
            "suffix": f"{active_contract_month_for_test}_15min_actual", 
            "title": f"N225先物 {active_contract_month_for_test} 15分足 (実データテスト)"
        },
        {
            "filename": f"n225_ohlc_{active_contract_month_for_test}_4H.csv", 
            "suffix": f"{active_contract_month_for_test}_4H_actual", 
            "title": f"N225先物 {active_contract_month_for_test} 4時間足 (実データテスト)"
        },
        # 他の足種や契約月のファイルがあれば追加
    ]

    for csv_info in test_csv_files_info:
        csv_file_name = csv_info["filename"]
        csv_path = os.path.join(ACTUAL_CSV_DIR, csv_file_name)

        if not os.path.exists(csv_path):
            print(f"SKIPPING: CSV file not found for test: {csv_path}")
            continue

        output_image_filename = f"chart_{csv_info['suffix']}.jpg"
        output_image_path = os.path.join(TEST_IMAGE_DIR, output_image_filename)
        chart_title = csv_info["title"]

        print(f"\n--- Testing with actual CSV: {csv_path} ---")
        print(f"Output image will be: {output_image_path}")
        print(f"Chart title: {chart_title}")
        
        try:
            generated_image_path = create_chart(
                csv_file=csv_path,
                output_file=output_image_path,
                title=chart_title,
                dpi=150 
            )
            if generated_image_path and os.path.exists(generated_image_path):
                print(f"SUCCESS: Chart created at {generated_image_path}")
            else:
                print(f"FAILURE: Chart creation failed or file not found for {csv_path}")
        except Exception as e:
            print(f"ERROR during chart creation for {csv_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- create_rosoku.py direct execution test finished ---")
