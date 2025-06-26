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
import os
# このcofigで共通の値を設定してる
from config import ROSOKU, CONTRACT_MONTH, USE_VOLUME, TIME_STEP


# 設定
DISPLAY_POINTS_PAST = 3000
DAYS_TO_PREDICT = 0.25

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
# スケーラーのファイル名を訓練スクリプトと合わせる
scaler_ohlcv_path = os.path.join(base_dir, f'saved_model_sakimono_{ROSOKU}/scaler_ohlcv_sakimono.pkl')
scaler_close_path = os.path.join(base_dir, f'saved_model_sakimono_{ROSOKU}/scaler_close_sakimono.pkl')

if not os.path.exists(model_save_path):
    print(f"エラー: モデルファイルが見つかりません: {model_save_path}")
    exit()
if not os.path.exists(scaler_ohlcv_path):
    print(f"エラー: 全特徴量スケーラーファイルが見つかりません: {scaler_ohlcv_path}")
    exit()
if not os.path.exists(scaler_close_path):
    print(f"エラー: 終値専用スケーラーファイルが見つかりません: {scaler_close_path}")
    exit()

model = load_model(model_save_path)
print(f"Model loaded from {model_save_path}")

with open(scaler_ohlcv_path, 'rb') as f:
    scaler_ohlcv = pickle.load(f) # 全特徴量スケーラー
print(f"OHLCV Scaler loaded from {scaler_ohlcv_path}")
with open(scaler_close_path, 'rb') as f:
    scaler_close = pickle.load(f) # 終値専用スケーラー
print(f"Close Scaler loaded from {scaler_close_path}")

# ===== データの読み込みと前処理 =====
# ROSOKU 変数に基づいてCSVファイルパスを動的に設定
base_csv_dir = os.path.join(base_dir, '..', 'chart_csvs') # CSVファイルがあるディレクトリ

csv_filename = f'n225_ohlc_{CONTRACT_MONTH}_{ROSOKU}.csv'

csv_file_path = os.path.join(base_csv_dir, csv_filename)
print(f"Using data from: {csv_file_path}")


if not os.path.exists(csv_file_path):
    print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
    exit()

df = pd.read_csv(csv_file_path, encoding='utf-8')
print(f"読み込んだデータ数: {len(df)}")

df.rename(columns={
    'Date': 'timestamp', 'Open': 'open', 'High': 'high',
    'Low': 'low', 'Close': 'close', 'Volume': 'volume'
}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# --- 使用する特徴量を「close」のみに固定 ---
feature_columns = ['close']
print("Closeのみで予測を実行します。")
print(f"予測に使用する特徴量: {feature_columns}")

data_for_scaling = df[feature_columns].values

# 保存済みスケーラー (scaler_ohlcv) を使用してデータを正規化
try:
    data_scaled = scaler_ohlcv.transform(data_for_scaling)
except ValueError as e:
    print(f"エラー: スケーラーの適用に失敗しました。訓練時と特徴量数が異なる可能性があります。エラー詳細: {e}")
    print(f"期待される特徴量数 (スケーラー): {scaler_ohlcv.n_features_in_}")
    print(f"現在のデータの特徴量数: {data_for_scaling.shape[1]}")
    exit()

close_idx = 0  # feature_columnsが['close']なのでインデックスは0


# ===== 時系列データセット作成関数 (入力は全特徴量、出力はスケーリング済み終値) =====
def create_sequences_for_prediction(input_data, time_step=1):
    """
    予測用の入力シーケンスのみを作成する (yは不要)
    input_data: スケーリング済みの全特徴量データ
    """
    dataX = []
    if len(input_data) < time_step: # 修正: time_step以上ないとシーケンス作れない
        print(f"警告: データセットの長さ ({len(input_data)}) が time_step ({time_step}) 未満です。")
        return np.array(dataX)
    for i in range(len(input_data) - time_step + 1): # 修正: +1 して最後のシーケンスまで含める
        a = input_data[i:(i + time_step), :] # 全特徴量
        dataX.append(a)
    return np.array(dataX)

# ===== パラメータ設定 =====
time_step = TIME_STEP  # 訓練時と同じ値

# --- ★復活: 過去データ全体に対する予測 ---
# 全データから予測用シーケンスを作成 (Xのみ)
X_all_sequences = create_sequences_for_prediction(data_scaled, time_step)

if X_all_sequences.size == 0:
    print("エラー: データセットからシーケンスを作成できませんでした。")
    exit()

# X_all_sequences は既に正しい形状 (サンプル数, time_step, 特徴量数) になっているはず
print("データ準備完了。過去データに対する予測を開始します...")

all_predict_scaled = model.predict(X_all_sequences, verbose=1) # 形状は (サンプル数, 1)
# 予測結果 (スケーリング済み終値) を元のスケールに戻す (終値専用スケーラーを使用)
all_predict = scaler_close.inverse_transform(all_predict_scaled)

# 比較対象の実際の終値 (y_all_actual) を準備
# create_dataset と同様のロジックで、スケーリング前のdfから実際の終値を取得し、
# all_predict と同じ数だけ用意する
actual_close_values_for_rmse = []
for i in range(len(data_scaled) - time_step):
    actual_close_values_for_rmse.append(df['close'].iloc[i + time_step])

y_all_actual = np.array(actual_close_values_for_rmse)

# RMSE計算のために長さを合わせる
if len(all_predict) == len(y_all_actual) + 1:
    all_predict_for_rmse = all_predict[:-1]
elif len(all_predict) == len(y_all_actual):
    all_predict_for_rmse = all_predict
else:
    print("RMSE計算のための実データと予測データの数が一致しません。")
    all_predict_for_rmse = None

if all_predict_for_rmse is not None and len(y_all_actual) == len(all_predict_for_rmse):
    overall_rmse = math.sqrt(mean_squared_error(y_all_actual, all_predict_for_rmse))
    print(f"全データに対するRMSE: {overall_rmse}")
# --- ★復活ここまで ---


ohlc_df = df.set_index('timestamp')

# ===== 未来の予測 =====
print("Predicting future steps...")
# 未来予測の最初の入力も全特徴量を使用
future_input_scaled = data_scaled[-time_step:, :].copy()

future_predictions_scaled = []
future_predictions_actual = []
future_dates = []

last_date = df['timestamp'].iloc[-1]
# current_sequence の形状も (1, time_step, 特徴量数) にする
current_sequence = future_input_scaled.reshape(1, time_step, data_scaled.shape[1])


# 予測する日数
days_to_predict = DAYS_TO_PREDICT

if time_delta_unit == "minutes":
    # 1時間あたりのステップ数
    steps_per_hour = 60 // time_delta_value
    # 1日あたりのステップ数
    steps_per_day = 24 * steps_per_hour
    num_future_steps = steps_per_day * days_to_predict # ここではまだfloatの可能性
elif time_delta_unit == "hours":
    # 1日あたりのステップ数
    steps_per_day = 24 // time_delta_value
    if steps_per_day == 0: steps_per_day = 1
    num_future_steps = steps_per_day * days_to_predict # ここではまだfloatの可能性
    if num_future_steps == 0 : num_future_steps = 1
else: # 日足の場合
    num_future_steps = days_to_predict # ここではまだfloatの可能性

# --- ここで整数に変換 ---
num_future_steps = int(num_future_steps)
if num_future_steps < 1: # 念のため、最低1ステップは予測するように
    num_future_steps = 1

print(f"Predicting {num_future_steps} steps ({time_delta_value} {time_delta_unit} interval, approx. {days_to_predict} days) into the future...")

for i in range(num_future_steps): # ここで整数が渡されるようにする
    future_pred_scaled_single = model.predict(current_sequence, verbose=0) # 出力は (1,1)
    future_predictions_scaled.append(future_pred_scaled_single[0,0]) # スケーリングされた値を保存

    # --- 暫定対応: 予測された終値で入力シーケンスの終値部分を更新し、他は最新値を維持 ---
    new_ohlcv_scaled_row = current_sequence[0, -1, :].copy() # 最新の行をコピー
    new_ohlcv_scaled_row[close_idx] = future_pred_scaled_single[0,0] # 終値部分を予測値で更新

    # current_sequence の更新
    new_sequence_rows = current_sequence[0, 1:, :]
    current_sequence = np.vstack((new_sequence_rows, new_ohlcv_scaled_row.reshape(1, -1)))
    current_sequence = current_sequence.reshape(1, time_step, data_scaled.shape[1])

    # 日付を進める処理をここに追加
    if time_delta_unit == "minutes":
        next_date = last_date + pd.Timedelta(minutes=time_delta_value * (i + 1))
    elif time_delta_unit == "hours":
        next_date = last_date + pd.Timedelta(hours=time_delta_value * (i + 1))
    else: # フォールバック
        next_date = last_date + pd.Timedelta(minutes=15 * (i + 1)) # デフォルトは15分

    future_dates.append(next_date)

# スケーリングされた未来予測をまとめて元のスケールに戻す (終値専用スケーラー)
future_predictions_actual = scaler_close.inverse_transform(np.array(future_predictions_scaled).reshape(-1,1)).flatten()

future_df = pd.DataFrame({'price': future_predictions_actual}, index=future_dates)
# 出力するCSVファイル名に ROSOKU を含める
future_csv_filename = f'future_predictions_sakimono_{ROSOKU}.csv'
future_csv_path = os.path.join(base_dir, future_csv_filename) # 保存先もスクリプト基準に
future_df.to_csv(future_csv_path, index_label='timestamp')
print(f"未来予測データをCSVに保存しました: {future_csv_path}")

# ===== 予測と実データの接続 =====
# 最後の実データ（ローソク足）の終値とタイムスタンプを取得
last_real_close = df['close'].iloc[-1]
last_real_timestamp = df['timestamp'].iloc[-1]

# --- ここから修正 ---
# 青線（過去データ予測）の最後の値とタイムスタンプを取得
if all_predict_for_rmse is not None and len(all_predict_for_rmse) > 0:
    last_pred_close = all_predict_for_rmse[-1]
    last_pred_timestamp = df['timestamp'].iloc[time_step + len(all_predict_for_rmse) - 1]
else:
    last_pred_close = last_real_close
    last_pred_timestamp = last_real_timestamp

# 予測リストの先頭に「青線の最後の予測値」を追加して、チャートの線を滑らかに接続
# last_pred_closeが配列の場合はfloatに変換
if isinstance(last_pred_close, (np.ndarray, list)):
    last_pred_close = float(np.array(last_pred_close).flatten()[0])
anchored_predictions = [last_pred_close] + list(np.array(future_predictions_actual).flatten())
anchored_dates = [last_pred_timestamp] + future_dates
# --- ここまで修正 ---

# ===== プロット用のデータ拡張 =====
# 注意: ここでのデータ拡張はチャート表示用であり、CSV保存される予測値とは異なります
for date_val, price_val in zip(anchored_dates, anchored_predictions):
    if date_val not in ohlc_df.index:
        # 新しい行のデータを作成
        new_row_data = {
            'Open': price_val,
            'High': price_val,
            'Low': price_val,
            'Close': price_val,
            'Volume': 0  # 予測なので出来高は0
        }
        # ohlc_dfに新しい行を追加
        ohlc_df.loc[date_val] = new_row_data

# ===== 結果のプロット =====
print("Drawing candlestick chart...")
display_points_past = DISPLAY_POINTS_PAST
# 拡張されたohlc_dfからサブセットを取得
ohlc_df_subset = ohlc_df.iloc[-(display_points_past + len(anchored_predictions) + 10):].copy()

# --- ★復活: 過去データ予測のプロット用Seriesを作成 ---
# predictPlot は終値のプロットなので、all_predict (逆スケーリング済み終値) を使用
predictPlot = np.empty((len(df), 1))
predictPlot[:, :] = np.nan
if all_predict_for_rmse is not None and len(all_predict_for_rmse) > 0:
    # flattenして1次元に
    all_predict_for_rmse_flat = np.array(all_predict_for_rmse).flatten()
    start_index_plot = time_step
    end_index_plot = start_index_plot + len(all_predict_for_rmse_flat)
    if end_index_plot <= len(predictPlot):
         predictPlot[start_index_plot:end_index_plot, 0] = all_predict_for_rmse_flat
    else:
        predictPlot[start_index_plot:, 0] = all_predict_for_rmse_flat[:len(predictPlot)-start_index_plot]
# プロット用のSeriesに変換
historical_pred_series = pd.Series(predictPlot.flatten(), index=df['timestamp'])
plot_historical_pred_series = ohlc_df_subset.index.map(lambda x: historical_pred_series.get(x, np.nan))
plot_historical_pred_series = pd.Series(plot_historical_pred_series, index=ohlc_df_subset.index)
# --- ★復活ここまで ---


# 変更: 接続されたデータでプロット用Seriesを作成
future_series_for_plot = pd.Series(anchored_predictions, index=anchored_dates)
plot_future_series = ohlc_df_subset.index.map(lambda x: future_series_for_plot.get(x, np.nan))
plot_future_series = pd.Series(plot_future_series, index=ohlc_df_subset.index)

apds = []
# --- ★復活: 過去データ予測のプロット追加処理 ---
if not plot_historical_pred_series.isna().all():
    apds.append(mpf.make_addplot(plot_historical_pred_series, type='line', color='blue', width=1.5, panel=0))
# --- ★復活ここまで ---

if not plot_future_series.isna().all():
    apds.append(mpf.make_addplot(plot_future_series, type='line', color='red', width=2.0, linestyle='--', panel=0, ylabel="Prediction"))

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