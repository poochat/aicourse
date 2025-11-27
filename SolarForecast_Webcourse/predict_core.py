# predict_core.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os

MODEL_PATHS = {
    "LSTM": "./saved_models/LSTM_best.keras",
    "GRU": "./saved_models/GRU_best.keras",
    "TCN": "./saved_models/TCN_best.keras",
    "Transformer": "./saved_models/Transformer_best.keras",
    "CNN-LSTM": "./saved_models/CNN-LSTM_best.keras"
}

models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models[name] = tf.keras.models.load_model(path)
            print(f"✅ 已加载模型：{name}")
        except Exception as e:
            print(f"❌ 模型 {name} 加载失败: {e}")
    else:
        print(f"⚠️ 警告：未找到模型文件 {path}")

# === 关键修复：正确加载并清洗数据 ===
DATA_PATH = "./data/processed/pv_2015_2016_경기도연성_clean.csv"

try:
    # 显式指定列名（与 app.py 一致）
    df_full = pd.read_csv(
        DATA_PATH,
        header=None,
        names=['year', 'month', 'day', 'hour', 'power_kW', 'poa_Wm2', 'ghi_Wm2', 'module_temp_C', 'ambient_temp_C', 'capacity_kW']
    )
    
    # 只保留数值特征列
    feature_cols = ['power_kW', 'poa_Wm2', 'ghi_Wm2', 'module_temp_C', 'ambient_temp_C']
    
    # 强制转换为 float，自动将无效值转为 NaN
    numeric_data = df_full[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # 删除包含 NaN 的行（可选，或用 fillna）
    numeric_data = numeric_data.dropna()
    
    if len(numeric_data) == 0:
        raise ValueError("清洗后无有效数值数据！")
    
    data_full = numeric_data.values  # shape: (N, 5)

    mean_vals = np.mean(data_full, axis=0)
    std_vals = np.std(data_full, axis=0)
    std_vals[std_vals == 0] = 1.0

    print("✅ 标准化参数计算成功！")

except Exception as e:
    print(f"❌ 数据加载或标准化失败: {e}")
    mean_vals = None
    std_vals = None

def predict_next_hour(model_name: str, latest_6hours):
    """
    latest_6hours: shape (6, 5), should be numeric
    Returns: predicted power in kW (float)
    """
    if model_name not in models or mean_vals is None:
        return None

    model = models[model_name]
    try:
        # 确保输入是 float
        input_array = np.array(latest_6hours, dtype=np.float32)
        norm_data = (input_array - mean_vals) / std_vals
        norm_data = norm_data.reshape(1, 6, 5)
        pred_norm = model.predict(norm_data, verbose=0)[0, 0]
        pred_kw = pred_norm * std_vals[0] + mean_vals[0]
        return round(float(pred_kw), 2)
    except Exception as e:
        print(f"预测出错: {e}")
        return None