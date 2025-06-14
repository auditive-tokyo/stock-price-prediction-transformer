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

# 設定 (このスクリプトの先頭あたりに追加)
ROSOKU = "15M"
# ROSOKU = "4H" # 4時間足を使用する場合はコメントアウトを外す

# ===== データの読み込みと前処理 =====
if ROSOKU == "15M":
    file_path = f'/Volumes/AUDITIVE/GitHub/stock-price-prediction-transformer/chart_csvs/n225_ohlc_202509_15min.csv'
elif ROSOKU == "4H":
    file_path = f'/Volumes/AUDITIVE/GitHub/stock-price-prediction-transformer/chart_csvs/n225_ohlc_202509_4H.csv'
else:
    print(f"エラー: 未対応のROSOKU設定です: {ROSOKU}")
    exit()
df = pd.read_csv(file_path)
print(f"Using data from: {file_path}")
print(f"読み込んだデータ数: {len(df)}")

# 'Date'列をdatetime型に変換し、ソート
df['Date'] = pd.to_datetime(df['Date']) # '日付' から 'Date' に変更
df = df.sort_values('Date') # '日付' から 'Date' に変更

# Close列のみを使用
data = df[['Close']].values  # '終値' から 'Close' に変更

# 'Close'用のスケーラー
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# ===== 時系列データセット作成関数 =====
def create_dataset(dataset, time_step=1):
    """
    時系列データをモデル入力用に変換する関数
    指定した時間ステップ数のシーケンスを作成し、次の値を予測対象とする
    """
    dataX, dataY = [], []
    if len(dataset) <= time_step:
        print(f"警告: データセットの長さ ({len(dataset)}) が time_step ({time_step}) 以下です。十分なデータがありません。")
        return np.array(dataX), np.array(dataY)
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ===== パラメータ設定とデータ分割 =====
time_step = 100  # 予測に使用する過去のデータポイント数 (データ量に応じて調整が必要)

# データを訓練用（80%）とテスト用（20%）に分割
training_size = int(len(data_scaled) * 0.80)
test_size = len(data_scaled) - training_size

if training_size <= time_step or test_size <= time_step:
    print(f"エラー: 訓練データサイズ ({training_size}) またはテストデータサイズ ({test_size}) が time_step ({time_step}) 以下です。")
    print("データ量を増やすか、time_stepを小さくしてください。")
    exit()

train_data, test_data = data_scaled[0:training_size,:], data_scaled[training_size:len(data_scaled),:]

# 訓練データとテストデータをモデル入力形式に変換
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# データセットが空でないか確認
if X_train.size == 0 or X_test.size == 0:
    print("エラー: 訓練データまたはテストデータが空です。create_datasetの処理結果を確認してください。")
    print(f"訓練データ入力形状: {X_train.shape}, テストデータ入力形状: {X_test.shape}")
    exit()

# ===== モデルのための入力データのリシェイプ =====
# [サンプル数, 時間ステップ, 特徴量(1)]の形式にデータを整形
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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
scaler_save_path = os.path.join(save_dir, 'scaler_sakimono.pkl') # os.path.join を使用
with open(scaler_save_path, 'wb') as f: 
    pickle.dump(scaler, f)
print(f"スケーラーを {scaler_save_path} に保存しました")

# ===== 予測の実行 =====
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# ===== 予測値をもとのスケールに戻す =====
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))


# ===== モデルの評価 =====
train_rmse = math.sqrt(mean_squared_error(y_train_orig, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test_orig, test_predict))

print(f"訓練データRMSE: {train_rmse}")
print(f"テストデータRMSE: {test_rmse}")