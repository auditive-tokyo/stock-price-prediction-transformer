# just an example yet
# This script is for zero-shot prediction using a pre-trained model.

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# 1. モデルとスケーラーをロード
model = load_model('saved_model_sakimono_15M/stock_transformer_model_sakimono.keras')
with open('saved_model_sakimono_15M/scaler_ohlcv_sakimono.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_model_sakimono_15M/scaler_close_sakimono.pkl', 'rb') as f:
    scaler_close = pickle.load(f)

# 2. 新しいデータを用意（例: DataFrame new_df）
# new_df = pd.read_csv('新しいデータ.csv')
# new_df['Date'] = pd.to_datetime(new_df['Date'])
# new_df = new_df.sort_values('Date')
feature_columns = ['Close']
data = new_df[feature_columns].values

# 3. スケーリング
data_scaled = scaler.transform(data)

# 4. time_step分のデータを使って入力を作成
time_step = 32  # 例
X_zero_shot = []
for i in range(len(data_scaled) - time_step + 1):
    X_zero_shot.append(data_scaled[i:i+time_step])
X_zero_shot = np.array(X_zero_shot)

# 5. 予測
pred_scaled = model.predict(X_zero_shot)
pred = scaler_close.inverse_transform(pred_scaled)