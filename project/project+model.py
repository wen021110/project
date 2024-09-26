import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Attention
from tensorflow.keras.saving import register_keras_serializable

##  資料預處理~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 載入資料集
file_path = './project - 資料集.csv'
df = pd.read_csv(file_path)


# 將“fish_date”轉換為日期時間格式
df['fish_date'] = pd.to_datetime(df['fish_date'], format='%Y/%m/%d %H:%M')

# 過濾資料以僅保留 7/12、7/18 和 7/31 的記錄
dates_to_keep = ['2024-07-12', '2024-07-18', '2024-07-31']
df_filtered = df[df['fish_date'].dt.strftime('%Y-%m-%d').isin(dates_to_keep)]

# 刪除這三天的前30筆資料
df_result = pd.DataFrame()

for date in dates_to_keep:
    date_filtered = df_filtered[df_filtered['fish_date'].dt.strftime('%Y-%m-%d') == date]
    date_filtered = date_filtered.iloc[30:]  
    df_result = pd.concat([df_result, date_filtered])

# 刪除空值
df_result = df_result.dropna(axis=1, how='all')
df_result = df_result.dropna()

# 計算ph的平均值
ph_mean = round(df_result[df_result['ph'] != 0]['ph'].mean(), 2)

# 取代ph=0的值，取代為平均值
df_result['ph'] = df_result['ph'].replace(0, ph_mean)

# 刪除溫度 <= 0 的欄位
df_result = df_result[df_result['tp'] > 0]

# 確保索引是日期時間格式
df_result.set_index('fish_date', inplace=True)

# 新增 'target' 欄位並根據日期填充值
df_result['target'] = np.where(df_result.index.strftime('%Y-%m-%d') == '2024-07-12', 0.85,
                       np.where(df_result.index.strftime('%Y-%m-%d') == '2024-07-18', 0.9,
                    np.where(df_result.index.strftime('%Y-%m-%d') == '2024-07-31', 1, np.nan)))

df_result.to_csv('test1.csv', index=False)

# 重置索引
df_result.reset_index(inplace=True)

# 分割資料集
X = df_result[['tp', 'ph', 'do', 'no2']].values
y = df_result['target'].values

# 資料標準化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 檢查資料集長度
print(f"Dataset length: {len(X_scaled)}")

# 將數據轉換為時間序列格式
sequence_length = 30

# 轉換為時間序列格式
X_sequences = []
y_sequences = []

for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i+sequence_length])
    y_sequences.append(y[i+sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# 檢查是否數據集大小匹配
assert len(X_sequences) == len(y_sequences), "X_sequences 和 y_sequences 的樣本數不一致！"

# 檢查形狀
print(f"X_sequences shape: {X_sequences.shape}")
print(f"y_sequences shape: {y_sequences.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2
                                                    , random_state=42)

# 檢查分割後的形狀
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 雙重注意力層
class DualAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DualAttention, self).__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        # inputs: [encoder_outputs, decoder_outputs]
        encoder_outputs, decoder_outputs = inputs

        # 計算注意力權重
        context_vector_1, attention_weights_1 = self.attention([encoder_outputs, encoder_outputs], return_attention_scores=True)
        context_vector_2, attention_weights_2 = self.attention([decoder_outputs, decoder_outputs], return_attention_scores=True)

        return context_vector_1 + context_vector_2

# 雙重注意力層，使用裝飾器註冊自定義層
@register_keras_serializable(package="Custom", name="DualAttention")
class DualAttention(Layer):
    def __init__(self, **kwargs):
        super(DualAttention, self).__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        encoder_outputs, decoder_outputs = inputs

        # 計算注意力權重
        context_vector_1, attention_weights_1 = self.attention([encoder_outputs, encoder_outputs], return_attention_scores=True)
        context_vector_2, attention_weights_2 = self.attention([decoder_outputs, decoder_outputs], return_attention_scores=True)

        return context_vector_1 + context_vector_2

# 模型建構
sequence_input = Input(shape=(sequence_length, X.shape[1]))
lstm_out = LSTM(64, return_sequences=True)(sequence_input)
lstm_out = LSTM(64, return_sequences=True)(lstm_out)

# 注意力層
attention_out = DualAttention()([lstm_out, lstm_out])

# GlobalAveragePooling1D 層將時間步維度壓縮
pooled_out = GlobalAveragePooling1D()(attention_out)

# 最後的輸出層
dense_out = Dense(1)(pooled_out)

# 建立模型
model = Model(inputs=sequence_input, outputs=dense_out)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 訓練模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 評估模型
y_pred = model.predict(X_test).flatten()

# 檢查預測結果形狀
print(f"y_pred shape: {y_pred.shape}")

# 計算績效指標
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 保存scaler
joblib.dump(scaler, 'scaler.save')

# 保存模型
model.save('fish_prediction_model.keras')