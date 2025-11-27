# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import numpy as np
import os
from predict_core import predict_next_hour, models

app = FastAPI(title="光伏发电功率智能预测系统")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATA_PATH = "./data/processed/pv_2015_2016_경기도연성_clean.csv"

# 加载并清洗数据
try:
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=['year', 'month', 'day', 'hour', 'power_kW', 'poa_Wm2', 'ghi_Wm2',
               'module_temp_C', 'ambient_temp_C', 'capacity_kW']
    )
    
    # 构造 datetime，自动处理类型
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')
    # 只保留有效时间行
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    
    # 转换数值列
    numeric_cols = ['power_kW', 'poa_Wm2', 'ghi_Wm2', 'module_temp_C', 'ambient_temp_C', 'capacity_kW']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✅ 成功加载 {len(df)} 行光伏数据")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    df = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, model: str = "LSTM"):
    if df is None or len(df) < 7:
        return "<h1 style='color:white;text-align:center;margin-top:50px;'>❌ 数据不足或加载失败！</h1>"
    
    if model not in models:
        model = "LSTM"

    latest_data = df.tail(7).copy()
    if len(latest_data) < 7:
        return "<h1 style='color:white;text-align:center;margin-top:50px;'>数据不足，无法预测！</h1>"

    # === 修复 FutureWarning：用 np.array 包裹 to_pydatetime() ===
    datetimes = np.array(latest_data['datetime'].dt.to_pydatetime())
    is_continuous = True
    for i in range(1, len(datetimes)):
        diff = (datetimes[i] - datetimes[i-1]).total_seconds()
        if diff != 3600:  # 不是正好1小时
            is_continuous = False
            break

    if not is_continuous:
        return "<h1 style='color:white;text-align:center;margin-top:50px;'>最近7小时数据不连续（非每小时记录），无法预测！</h1>"

    # 提取输入数据（确保是 float）
    input_data = latest_data.iloc[:-1][['power_kW','poa_Wm2','ghi_Wm2','module_temp_C','ambient_temp_C']].values
    if input_data.shape != (6, 5):
        return "<h1 style='color:white;text-align:center;margin-top:50px;'>输入数据维度错误！</h1>"

    pred_power = predict_next_hour(model, input_data)
    if pred_power is None or pred_power != pred_power:  # NaN check
        return "<h1 style='color:white;text-align:center;margin-top:50px;'>模型预测返回无效值！</h1>"

    current_row = latest_data.iloc[-1]
    current_time = current_row['datetime'].strftime("%Y-%m-%d %H:%M")
    current_hour = current_row['datetime'].hour
    next_hour = (current_hour + 1) % 24

    # === 关键修复：确保 recent_data 中的 hour 是 int 类型 ===
    latest_6_rows = latest_data.iloc[:-1].copy()
    latest_6_rows['hour'] = pd.to_numeric(latest_6_rows['hour'], errors='coerce')
    latest_6_rows = latest_6_rows.dropna(subset=['hour'])
    latest_6_rows['hour'] = latest_6_rows['hour'].astype(int)

    recent_6 = latest_6_rows[['hour','power_kW','poa_Wm2','module_temp_C']].round(2).to_dict('records')

    history_df = df.tail(48)[['datetime', 'power_kW']].dropna()
    history_labels = [dt.strftime("%H:%M") for dt in history_df['datetime']]
    history_values = history_df['power_kW'].round(2).tolist()

    capacity = int(df['capacity_kW'].iloc[0])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_time": current_time,
        "next_hour": next_hour,
        "pred_power": pred_power,
        "current_power": round(current_row['power_kW'], 2),
        "capacity": capacity,
        "recent_data": recent_6,
        "history_labels": history_labels,
        "history_values": history_values,
        "selected_model": model,
        "available_models": list(models.keys())
    })

if __name__ == "__main__":
    print("光伏发电功率预测系统启动成功！")
    print("访问地址：http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)