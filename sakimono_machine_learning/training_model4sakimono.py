import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error
import math
import os
import pickle
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
# このcofigで共通の値を設定してる
from config import ROSOKU, CONTRACT_MONTH, USE_VOLUME, TIME_STEP


# ===== データの読み込みと前処理 =====
if ROSOKU == "15M":
    file_path = f'/Volumes/AUDITIVE/GitHub/stock-price-prediction-transformer/chart_csvs/n225_ohlc_{CONTRACT_MONTH}_{ROSOKU}.csv'
elif ROSOKU == "4H":
    file_path = f'/Volumes/AUDITIVE/GitHub/stock-price-prediction-transformer/chart_csvs/n225_ohlc_{CONTRACT_MONTH}_{ROSOKU}.csv'
# elif ROSOKU == "1D":
    # file_path = f'/Volumes/AUDITIVE/GitHub/stock-price-prediction-transformer/chart_csvs/n225_ohlc_{CONTRACT_MONTH}_1D.csv'
else:
    print(f"エラー: 未対応のROSOKU設定です: {ROSOKU}")
    exit()
df = pd.read_csv(file_path)
print(f"Using data from: {file_path}")
print(f"読み込んだデータ数: {len(df)}")

# 'Date'列をdatetime型に変換し、ソート
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- 使用する特徴量を USE_VOLUME フラグに基づいて設定 ---
if USE_VOLUME:
    if 'Volume' in df.columns:
        print("Volumeを含めて学習します。")
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        print("警告: 'Volume' 列が見つかりませんが、USE_VOLUMEがTrueです。OHLCのみを使用します。")
        feature_columns = ['Open', 'High', 'Low', 'Close']
else:
    print("Volumeを含めずにOHLCのみで学習します。")
    feature_columns = ['Open', 'High', 'Low', 'Close']


data = df[feature_columns].values

# 特徴量スケーラー
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 予測対象である終値（Close）のスケーリングを別途行うためのスケーラー
# Close列が何番目の特徴量かを取得 (通常は3番目、0-indexed)
close_idx = feature_columns.index('Close')
scaler_close = MinMaxScaler(feature_range=(0,1))
# data_scaledからClose列だけを取り出して、scaler_closeをfitする
# 注意: ここでfit_transformするとdata_scaledの値が変わってしまうので、
# df['Close']の元の値でfitするか、data_scaledの該当列でfitする。
# ここでは、元のCloseデータでfitする。
scaled_close_for_y = scaler_close.fit_transform(df[['Close']].values)


# ===== 時系列データセット作成関数 =====
def create_dataset(dataset_x, dataset_y_target_scaled, time_step=1):
    """
    時系列データをモデル入力用に変換する関数
    dataset_x: 入力特徴量 (スケーリング済み、複数特徴量)
    dataset_y_target_scaled: 予測対象の終値 (スケーリング済み、1特徴量)
    """
    dataX, dataY = [], []
    if len(dataset_x) <= time_step:
        print(f"警告: データセットの長さ ({len(dataset_x)}) が time_step ({time_step}) 以下です。")
        return np.array(dataX), np.array(dataY)
    # ループの範囲を修正: len(dataset_x) - time_step で十分
    for i in range(len(dataset_x) - time_step):
        # 入力シーケンス (全特徴量)
        a = dataset_x[i:(i + time_step), :] # 全特徴量を使用
        dataX.append(a)
        # 予測対象 (次の時間ステップのスケーリング済み終値)
        dataY.append(dataset_y_target_scaled[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ===== パラメータ設定とデータ分割 =====
time_step = TIME_STEP

# データを訓練用（80%）とテスト用（20%）に分割
training_size = int(len(data_scaled) * 0.80)
test_size = len(data_scaled) - training_size

if training_size <= time_step or test_size <= time_step:
    print(f"エラー: 訓練データサイズ ({training_size}) またはテストデータサイズ ({test_size}) が time_step ({time_step}) 以下です。")
    exit()

# 入力特徴量データ
train_data_x, test_data_x = data_scaled[0:training_size,:], data_scaled[training_size:len(data_scaled),:]
# 予測対象データ (スケーリング済み終値)
train_data_y, test_data_y = scaled_close_for_y[0:training_size,:], scaled_close_for_y[training_size:len(scaled_close_for_y),:]


# 訓練データとテストデータをモデル入力形式に変換
X_train, y_train = create_dataset(train_data_x, train_data_y, time_step)
X_test, y_test = create_dataset(test_data_x, test_data_y, time_step)


if X_train.size == 0 or X_test.size == 0:
    print("エラー: 訓練データまたはテストデータが空です。")
    exit()

# ===== モデルのための入力データのリシェイプ =====
# X_train, X_test は既に create_dataset で正しい形状になっているはず
# [サンプル数, 時間ステップ, 特徴量数]
# 特徴量数は data_scaled.shape[1]

# ===== Transformerエンコーダーブロックの定義 =====
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Transformerエンコーダーブロックを定義
    - head_size: 各アテンションヘッドの次元数
    - num_heads: マルチヘッドアテンションのヘッド数
    - ff_dim: フィードフォワードネットワークの次元数
    - dropout: ドロップアウト率
    """
    # 第1のサブレイヤー: マルチヘッドセルフアテンション
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # 第2のサブレイヤー: フィードフォワードネットワーク
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# ===== モデル構築 =====
inputs = Input(shape=(X_train.shape[1], X_train.shape[2])) 
# --- !!! 注意: データ量が少ない場合、モデルの複雑さ（層の数、ユニット数）を減らすことを検討してください !!! ---
x = transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.1) # パラメータ調整例
x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=64, dropout=0.15) # パラメータ調整例

x = GlobalAveragePooling1D(data_format='channels_last')(x)
x = Dropout(0.15)(x) 
x = Dense(10, activation="relu", kernel_regularizer=l2(1e-4))(x) # パラメータ調整例
outputs = Dense(1, activation="linear", kernel_regularizer=l2(1e-4))(x) 

model = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(learning_rate=0.001) # 学習率調整例
model.compile(optimizer=optimizer, loss="mean_squared_error")

model.summary()

# ===== モデルの訓練 =====
# --- !!! 注意: データ量が少ない場合、patience や epochs を調整してください !!! ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1) # patience調整例
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00005, verbose=1) # patience調整例

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, verbose=1, callbacks=[early_stopping, reduce_lr]) # batch_size, epochs調整例

# 保存ディレクトリの確認と作成
# このスクリプトファイルがあるディレクトリを取得
base_script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_script_dir, f'saved_model_sakimono_{ROSOKU}')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"ディレクトリ {save_dir} を作成しました")

# モデルを保存
model_save_path = os.path.join(save_dir, 'stock_transformer_model_sakimono.keras') # os.path.join を使用
model.save(model_save_path)
print(f"モデルを {model_save_path} に保存しました")

# 学習履歴を保存
history_dict = history.history
history_save_path = os.path.join(save_dir, 'training_history_sakimono.npy') # os.path.join を使用
np.save(history_save_path, history_dict)
print(f"学習履歴を {history_save_path} に保存しました")

# スケーラーも保存しておく（予測時に同じスケーリングを適用するため）
# scaler は全特徴量用、scaler_close は終値専用
scaler_save_path = os.path.join(save_dir, 'scaler_ohlcv_sakimono.pkl') # 名前を変更
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"全特徴量スケーラーを {scaler_save_path} に保存しました")

scaler_close_save_path = os.path.join(save_dir, 'scaler_close_sakimono.pkl')
with open(scaler_close_save_path, 'wb') as f:
    pickle.dump(scaler_close, f)
print(f"終値専用スケーラーを {scaler_close_save_path} に保存しました")


# ===== 予測の実行 =====
train_predict_scaled = model.predict(X_train)
test_predict_scaled = model.predict(X_test)

# ===== 予測値をもとのスケールに戻す (終値専用スケーラーを使用) =====
train_predict = scaler_close.inverse_transform(train_predict_scaled)
test_predict = scaler_close.inverse_transform(test_predict_scaled)

# y_train, y_test は既にスケーリングされた終値なので、これらも逆変換する
y_train_orig = scaler_close.inverse_transform(y_train.reshape(-1, 1))
y_test_orig = scaler_close.inverse_transform(y_test.reshape(-1, 1))


# ===== モデルの評価 =====
train_rmse = math.sqrt(mean_squared_error(y_train_orig, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test_orig, test_predict))

print(f"訓練データRMSE: {train_rmse}")
print(f"テストデータRMSE: {test_rmse}")