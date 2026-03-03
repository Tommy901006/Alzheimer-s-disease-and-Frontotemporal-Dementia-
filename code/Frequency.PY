import pandas as pd
import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import simpson

# 頻段設定
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 25),
    'Gamma': (25, 45)
}

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)

def band_power(freqs, psd, band):
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return float(np.trapz(psd[mask], freqs[mask]))
    return float(simpson(y=psd[mask], x=freqs[mask]))

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="🧠 EEG 頻段分析工具", layout="wide")
st.title("🧠 EEG 頻段分析工具 (Streamlit 版)")

# Sidebar parameters
st.sidebar.header("參數設定")
fs = st.sidebar.number_input("取樣率 (Hz)", value=500, step=1)
win_sec = st.sidebar.number_input("視窗長度 (秒)", value=4.0, step=0.5)
overlap_pct = st.sidebar.slider("重疊百分比 (%)", 0, 90, 0, step=5)
use_percentage = st.sidebar.checkbox("相對功率使用百分比 (%)", value=True)

# File upload
uploaded_files = st.file_uploader("上傳 CSV 檔案 (可多選)", type=["csv"], accept_multiple_files=True)

results = []

if uploaded_files:
    window_size = int(win_sec * fs)
    step_size = int(window_size * (1 - overlap_pct / 100))

    for file in uploaded_files:
        st.subheader(f"處理檔案：{file.name}")
        try:
            df = pd.read_csv(file)
            time_column = None
            if df.columns[0].lower() in ["time", "timestamp"]:
                time_column = df.columns[0]
                time_data = df[time_column]
                df = df.drop(columns=[time_column])

            # 使用者選擇要分析的欄位
            selected_cols = st.multiselect(
                f"選擇要分析的欄位 ({file.name})",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )

            n = len(df)
            relative_power_table = []

            for ch in selected_cols:
                x = df[ch].values.astype(float)
                band_power_list = {b: [] for b in bands}
                for start in range(0, n - window_size + 1, step_size):
                    segment = x[start:start + window_size]
                    freqs, psd = welch(segment, fs=fs, nperseg=window_size)
                    for band, (lo, hi) in bands.items():
                        bp = band_power(freqs, psd, (lo, hi))
                        band_power_list[band].append(bp)

                avg_power = {band: np.mean(band_power_list[band]) for band in bands}
                total_power = sum(avg_power.values())
                if use_percentage:
                    rel_power = {band: (p / total_power * 100) if total_power > 0 else 0 for band, p in avg_power.items()}
                else:
                    rel_power = {band: (p / total_power) if total_power > 0 else 0 for band, p in avg_power.items()}

                rel_power["Channel"] = ch
                relative_power_table.append(rel_power)

            # 相對功率表格
            rel_df = pd.DataFrame(relative_power_table)
            rel_df = rel_df[["Channel"] + list(bands.keys())]
            st.dataframe(rel_df.style.highlight_max(axis=0))

            # 攤平成單列格式
            flattened = {}
            for _, row in rel_df.iterrows():
                for band in bands:
                    flattened[f"{row['Channel']}_{band}"] = row[band]
            flat_df = pd.DataFrame([flattened])

            # 保存結果
            results.append({
                "File": file.name,
                "RelativePower": flat_df
            })

            st.success(f"✅ 完成：{file.name}")

        except Exception as e:
            st.error(f"❌ 錯誤：{file.name}：{str(e)}")

    # 彙整下載
    if results:
        st.subheader("📊 分析結果下載")
        all_flat = pd.concat([r["RelativePower"].assign(File=r["File"]) for r in results], ignore_index=True)
        st.dataframe(all_flat)

        csv = all_flat.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="下載所有結果 (CSV)",
            data=csv,
            file_name="EEG_relative_band_power_results.csv",
            mime="text/csv"
        )