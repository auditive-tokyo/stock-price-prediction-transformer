import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
import pickle
import os # osモジュールをインポート

# 設定
DISPLAY_POINTS_PAST = 1000
DAYS_TO_PREDICT = 2
ROSOKU = "15M"
# ROSOKU = "4H" # 4時間足を使用する場合はコメントアウトを外す
CONTRACT_MONTH = "202509" # 契約月も変数化しておくと便利

# --- データの足種に応じた時間間隔設定を ROSOKU から動的に設定 ---
if ROSOKU == "15M":
    time_delta_unit = "minutes"
    time_delta_value = 15
elif ROSOKU == "4H":
    time_delta_unit = "hours"
    time_delta_value = 4
else:
    print(f"警告: 未定義のROSOKU値 '{ROSOKU}' です。デフォルトの時間間隔を使用します (minutes, 15)。")
    time_delta_unit = "minutes"
    time_delta_value = 15

# ===== 保存済みのモデルとスケーラーを読み込む =====
base_dir = os.path.dirname(__file__)
model_save_path = os.path.join(base_dir, f'saved_model_sakimono_{ROSOKU}/stock_transformer_model_sakimono.keras')
scaler_path = os.path.join(base_dir, f'saved_model_sakimono_{ROSOKU}/scaler_sakimono.pkl')

if not os.path.exists(model_save_path):
    print(f"エラー: モデルファイルが見つかりません: {model_save_path}")
    exit()
if not os.path.exists(scaler_path):
    print(f"エラー: スケーラーファイルが見つかりません: {scaler_path}")
    exit()

model = load_model(model_save_path)
print(f"Model loaded from {model_save_path}")

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print(f"Scaler loaded from {scaler_path}")

# ===== データの読み込みと前処理 =====
# ROSOKU 変数に基づいてCSVファイルパスを動的に設定
base_csv_dir = os.path.join(base_dir, '..', 'chart_csvs') # CSVファイルがあるディレクトリ
csv_filename = ""
if ROSOKU == "15M":
    csv_filename = f'n225_ohlc_{CONTRACT_MONTH}_15min.csv'
elif ROSOKU == "4H":
    csv_filename = f'n225_ohlc_{CONTRACT_MONTH}_4H.csv'
else:
    print(f"エラー: 未対応のROSOKU設定です: {ROSOKU}。CSVファイルを特定できません。")
    exit()
csv_file_path = os.path.join(base_csv_dir, csv_filename)
print(f"Using data from: {csv_file_path}")


if not os.path.exists(csv_file_path):
    print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
    exit()

df = pd.read_csv(csv_file_path, encoding='utf-8') # エンコーディングを utf-8 に変更
print(f"読み込んだデータ数: {len(df)}")

# カラム名を英語の小文字に変更 (mplfinance との互換性のため、また元データに合わせて)
# 元のCSVヘッダーが 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' であることを想定
df.rename(columns={
    'Date': 'timestamp', # 'Date' を 'timestamp' に
    'Open': 'open',      # 'Open' を 'open' に
    'High': 'high',      # 'High' を 'high' に
    'Low': 'low',        # 'Low' を 'low' に
    'Close': 'close',    # 'Close' を 'close' に
    'Volume': 'volume'   # 'Volume' を 'volume' に
}, inplace=True)


# timestamp列を日付型に変換
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp') # 念のためソート

data = df[['close']].values  # close列のみを取得 (リネーム後のカラム名)

# 保存済みスケーラーを使用してデータを正規化
data_scaled = scaler.transform(data)

# ===== 時系列データセット作成関数 =====
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    if len(dataset) <= time_step:
        return np.array(dataX), np.array(dataY)
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ===== パラメータ設定 =====
# --- !!! 注意: time_step は訓練時と必ず同じ値にしてください !!! ---
time_step = 100  # 訓練時と同じ値に設定 (training_model4sakimono.py の値と合わせる)

X_all, y_all = create_dataset(data_scaled, time_step)

if X_all.size == 0:
    print("エラー: データセットからシーケンスを作成できませんでした。データ量とtime_stepを確認してください。")
    exit()

X_all_reshaped = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)
print("データ準備完了。予測を開始します...")

all_predict_scaled = model.predict(X_all_reshaped, verbose=1)
all_predict = scaler.inverse_transform(all_predict_scaled)
y_all_actual = scaler.inverse_transform(y_all.reshape(-1, 1))

if len(y_all_actual) == len(all_predict):
    overall_rmse = math.sqrt(mean_squared_error(y_all_actual, all_predict))
    print(f"全データに対するRMSE: {overall_rmse}")
else:
    print("RMSE計算のための実データと予測データの数が一致しません。")

# ===== 結果のプロット準備 =====
predictPlot = np.empty_like(data_scaled)
predictPlot[:, :] = np.nan
if len(all_predict) > 0:
    start_index_plot = time_step + 1
    end_index_plot = start_index_plot + len(all_predict)
    if end_index_plot <= len(predictPlot):
         predictPlot[start_index_plot:end_index_plot, :] = all_predict
    else:
        predictPlot[start_index_plot:, :] = all_predict[:len(predictPlot)-start_index_plot]

ohlc_df = df.set_index('timestamp')

# ===== 未来の予測 ===== # コメントを少し変更
print("Predicting future steps...") # メッセージを少し変更
future_input_scaled = data_scaled[-time_step:].copy()
future_predictions = []
future_dates = []
last_date = df['timestamp'].iloc[-1]
current_sequence = future_input_scaled.reshape(1, time_step, 1)


# 予測する日数
days_to_predict = DAYS_TO_PREDICT

if time_delta_unit == "minutes":
    # 1時間あたりのステップ数
    steps_per_hour = 60 // time_delta_value
    # 1日あたりのステップ数
    steps_per_day = 24 * steps_per_hour
    num_future_steps = steps_per_day * days_to_predict
elif time_delta_unit == "hours":
    # 1日あたりのステップ数
    steps_per_day = 24 // time_delta_value
    if steps_per_day == 0: steps_per_day = 1 # 時間足が24時間より大きい場合は1日1ステップとみなす（ありえないが念のため）
    num_future_steps = steps_per_day * days_to_predict
    if num_future_steps == 0 : num_future_steps = 1 # 少なくとも1ステップは予測
else: # 日足の場合
    num_future_steps = days_to_predict # 日足なら日数そのままがステップ数

print(f"Predicting {num_future_steps} steps ({time_delta_value} {time_delta_unit} interval, approx. {days_to_predict} days) into the future...")


for i in range(num_future_steps):
    future_pred_scaled_single = model.predict(current_sequence, verbose=0)
    future_pred_actual_single = scaler.inverse_transform(future_pred_scaled_single)[0][0]
    future_predictions.append(future_pred_actual_single)
    
    new_element_scaled = future_pred_scaled_single.flatten()
    current_sequence_np = current_sequence.flatten()
    current_sequence_np = np.append(current_sequence_np[1:], new_element_scaled)
    current_sequence = current_sequence_np.reshape(1, time_step, 1)

    # 日付を進める
    if time_delta_unit == "minutes":
        next_date = last_date + pd.Timedelta(minutes=time_delta_value * (i + 1))
    elif time_delta_unit == "hours":
        next_date = last_date + pd.Timedelta(hours=time_delta_value * (i + 1))
    elif time_delta_unit == "days":
        next_date = last_date + pd.Timedelta(days=time_delta_value * (i + 1))
    else: # デフォルトは日単位
        next_date = last_date + pd.Timedelta(days=1 * (i + 1))
    future_dates.append(next_date)

future_df = pd.DataFrame({'price': future_predictions}, index=future_dates)
# 出力するCSVファイル名に ROSOKU を含める
future_csv_filename = f'future_predictions_sakimono_{ROSOKU}.csv'
future_csv_path = os.path.join(base_dir, future_csv_filename) # 保存先もスクリプト基準に
future_df.to_csv(future_csv_path, index_label='timestamp')
print(f"未来予測データをCSVに保存しました: {future_csv_path}")

# ===== プロット用のデータ拡張 =====
for date_val, price_val in zip(future_dates, future_predictions):
    new_row = pd.DataFrame({
        'open': price_val, 'high': price_val, 'low': price_val, 'close': price_val, 'volume': 0
    }, index=[date_val])
    ohlc_df = pd.concat([ohlc_df, new_row])

# ===== 結果のプロット =====
print("Drawing candlestick chart...")
display_points_past = DISPLAY_POINTS_PAST
ohlc_df_subset = ohlc_df.iloc[-(display_points_past + 10):].copy()

actual_pred_dates = df['timestamp'].iloc[time_step+1 : time_step+1+len(all_predict)]
actual_pred_series = pd.Series(all_predict.flatten(), index=actual_pred_dates)
plot_pred_series = ohlc_df_subset.index.map(lambda x: actual_pred_series.get(x, np.nan))
plot_pred_series = pd.Series(plot_pred_series, index=ohlc_df_subset.index)

future_series_for_plot = pd.Series(future_predictions, index=future_dates)
plot_future_series = ohlc_df_subset.index.map(lambda x: future_series_for_plot.get(x, np.nan))
plot_future_series = pd.Series(plot_future_series, index=ohlc_df_subset.index)

apds = []
if not plot_pred_series.isna().all():
    apds.append(mpf.make_addplot(plot_pred_series, type='line', color='blue', width=1.5, panel=0, ylabel="Prediction")) # panel指定とylabel追加
if not plot_future_series.isna().all():
    apds.append(mpf.make_addplot(plot_future_series, type='line', color='red', width=2.0, linestyle='--', panel=0)) # panel指定

mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')

# 出力するチャート画像ファイル名に ROSOKU を含める
chart_filename_suffix = f"prediction_{ROSOKU}"
chart_save_path = os.path.join(base_dir, f'nikkei_futures_candlestick_{chart_filename_suffix}.jpg')

if len(apds) > 0:
    fig, axes = mpf.plot(ohlc_df_subset, type='candle', style=s,
                        title=f'Nikkei Futures Price Prediction (Transformer - {ROSOKU})', # タイトルにもROSOKU追加
                        ylabel='Price', addplot=apds, volume=True,
                        datetime_format='%Y-%m-%d %H:%M', xrotation=45,
                        figsize=(16, 9), returnfig=True, panel_ratios=(3,1))
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1.5, label='Historical Prediction'),
        Line2D([0], [0], color='red', lw=2.0, linestyle='--', label='Future Prediction')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig(chart_save_path, dpi=300)
    print(f"Candlestick chart saved as '{chart_save_path}'")
else:
    # 予測がない場合のチャートファイル名にもROSOKUを含める
    chart_only_filename_suffix = f"only_{ROSOKU}"
    chart_save_path_only = os.path.join(base_dir, f'nikkei_futures_candlestick_{chart_only_filename_suffix}.jpg')
    fig, axes = mpf.plot(ohlc_df_subset, type='candle', style=s,
            title=f'Nikkei Futures Price Chart ({ROSOKU})', # タイトルにもROSOKU追加
            ylabel='Price', volume=True,
            datetime_format='%Y-%m-%d %H:%M', xrotation=45,
            figsize=(16, 9), returnfig=True, panel_ratios=(3,1))
    plt.tight_layout()
    plt.savefig(chart_save_path_only, dpi=300)
    print(f"Candlestick chart saved as '{chart_save_path_only}'")

plt.close(fig)
print("Completed.")