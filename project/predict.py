import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 設定序列長度
sequence_length = 30

# 加載模型與scaler
model = load_model('fish_prediction_model.keras')
scaler = joblib.load('scaler.save')

file_path = './project - 資料集.csv'
new_data = pd.read_csv(file_path)

# 讀取時間戳列
timestamps = new_data['fish_date']

# 選擇數值型欄位，並將其轉換為浮點數類型
X = new_data[['tp', 'ph', 'do', 'no2']].astype(float)

def preprocess_new_data(X, scaler, sequence_length=30):
    # 將數據縮放
    X_scaled = scaler.transform(X)
    
    # 將數據轉換為時間序列格式
    new_sequences = []
    for i in range(len(X_scaled) - sequence_length):
        new_sequences.append(X_scaled[i:i+sequence_length])
    new_sequences = np.array(new_sequences)
    return new_sequences

# 預處理新數據
new_data_sequences = preprocess_new_data(X, scaler, sequence_length)

# 使用模型進行預測
predictions = model.predict(new_data_sequences)

# 用來存儲結果的列表
indicator_advice_list = []
prediction_advice_list = []

# 個別數值建議(人工設定範圍)
def give_advice_based_on_indicators(X, predictions, timestamps, sequence_length):
    # 定義合理範圍
    tp_threshold = (0, 30)  # tp 合理範圍
    ph_threshold = (4, 8.5)  # ph 合理範圍
    do_threshold = (1000, 3000)  # 溶解氧合理範圍
    no2_threshold = (0,800)  # 氮氧化物合理範圍

    # 檢查每一筆資料
    for i, (features, pred) in enumerate(zip(X.values, predictions)):
        tp, ph, do, no2 = features  # 直接從數組中解壓出特徵
        pred_value = pred[0]  # 獲取模型預測值

        # 判斷是否超出合理範圍，並僅當超出時添加警示
        warnings = []
        if not (tp_threshold[0] <= tp <= tp_threshold[1]):
            warnings.append(f"tp ({tp}) 超出合理範圍")
        if not (ph_threshold[0] <= ph <= ph_threshold[1]):
            warnings.append(f"ph ({ph}) 超出合理範圍")
        if not (do_threshold[0] <= do <= do_threshold[1]):
            warnings.append(f"溶解氧 do ({do}) 超出合理範圍")
        if not (no2_threshold[0] <= no2 <= no2_threshold[1]):
            warnings.append(f"氮氧化物 no2 ({no2}) 超出合理範圍")

        # 只有在有警示時才保存到結果列表
        if warnings:
            timestamp = timestamps.iloc[i]
            indicator_advice_list.append({
                'Sample': i + 1,
                'Time': timestamp,
                'Warnings': ', '.join(warnings),
                'Prediction': pred_value
            })

# 呼叫根據個別指標給出建議的函數
give_advice_based_on_indicators(X, predictions, timestamps, sequence_length)

# 建議函數
def give_advice(predictions, timestamps, sequence_length):
    for i, pred in enumerate(predictions):
        pred_value = pred[0]
        if pred_value > 0.9:
            advice = "建議保持當前操作。"
        elif pred_value > 0.8:
            advice = "建議監控水質變化。"
        else:
            advice = "建議檢查水質並適當調整。"
        
        timestamp = timestamps.iloc[i]
        prediction_advice_list.append({
            'Sample': i + 1,
            'Time': timestamp,
            'Prediction': pred_value,
            'Advice': advice
        })

# 呼叫根據整體預測值給出建議的函數
give_advice(predictions, timestamps, sequence_length)

# 列印結果
## 個別數據警示
print("Indicator Advice List:")
for advice in indicator_advice_list:
    print(f"Sample {advice['Sample']}: 時間 {advice['Time']}")
    print(f"  - 預測值: {advice['Prediction']:.2f}，{advice['Warnings']}")
## 整體數據警示
print("\nPrediction Advice List:")
for advice in prediction_advice_list:
    print(f"Sample {advice['Sample']}: 時間 {advice['Time']}")
    print(f"  - 預測值: {advice['Prediction']:.2f}，{advice['Advice']}")

# 將結果存儲到 DataFrame 並輸出到 Excel 文件
indicator_advice_df = pd.DataFrame(indicator_advice_list)
prediction_advice_df = pd.DataFrame(prediction_advice_list)

# 創建一個 Excel writer，將兩個建議輸出到不同的工作表
with pd.ExcelWriter('advice_output.xlsx') as writer:
    indicator_advice_df.to_excel(writer, sheet_name='個別數值警示', index=False)
    prediction_advice_df.to_excel(writer, sheet_name='預測警示', index=False)

