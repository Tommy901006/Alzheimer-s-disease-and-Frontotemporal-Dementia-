import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
    mask = np.logical_and(freqs >= band[0], freqs <= band[1])
    if np.sum(mask) >= 3:
        return simpson(psd[mask], freqs[mask])
    else:
        return np.trapz(psd[mask], freqs[mask])

class EEGAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 EEG 頻段分析工具")
        self.root.geometry("700x480")

        self.create_widgets()
        # === 加在 self.create_widgets 下方
        self.use_percentage = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.root, text="相對功率使用百分比 (%)", variable=self.use_percentage).pack(pady=2)

    def create_widgets(self):
        # 輸入資料夾
        ttk.Label(self.root, text="輸入資料夾：").pack(anchor="w", padx=10)
        self.input_entry = ttk.Entry(self.root, width=80)
        self.input_entry.pack(padx=10)
        ttk.Button(self.root, text="選擇資料夾", command=self.select_input_folder).pack(padx=10, pady=3)

        # 輸出資料夾
        ttk.Label(self.root, text="輸出資料夾：").pack(anchor="w", padx=10)
        self.output_entry = ttk.Entry(self.root, width=80)
        self.output_entry.pack(padx=10)
        ttk.Button(self.root, text="選擇資料夾", command=self.select_output_folder).pack(padx=10, pady=3)

        # 參數設定區
        param_frame = ttk.Frame(self.root)
        param_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(param_frame, text="取樣率 (Hz)：").grid(row=0, column=0, sticky="w")
        self.fs_entry = ttk.Entry(param_frame, width=10)
        self.fs_entry.insert(0, "500")
        self.fs_entry.grid(row=0, column=1)

        ttk.Label(param_frame, text="視窗長度 (秒)：").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.win_entry = ttk.Entry(param_frame, width=10)
        self.win_entry.insert(0, "4")
        self.win_entry.grid(row=0, column=3)

        ttk.Label(param_frame, text="重疊百分比 (%)：").grid(row=0, column=4, sticky="w", padx=(20, 0))
        self.ov_entry = ttk.Entry(param_frame, width=10)
        self.ov_entry.insert(0, "0")
        self.ov_entry.grid(row=0, column=5)

        # 執行按鈕與進度條
        ttk.Button(self.root, text="▶ 開始分析", command=self.analyze).pack(pady=10)
        self.progress = ttk.Progressbar(self.root, length=500, mode="determinate")
        self.progress.pack(pady=5)

        # 日誌區域
        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(padx=10, pady=5, fill="both")

    def log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.root.update()

    def select_input_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, path)

    def select_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def analyze(self):
        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()
        try:
            fs = int(self.fs_entry.get())
            win_sec = float(self.win_entry.get())
            overlap_pct = float(self.ov_entry.get())
        except:
            messagebox.showerror("錯誤", "請確認參數為數值")
            return

        files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        if not files:
            messagebox.showwarning("無檔案", "該資料夾內沒有 CSV 檔案！")
            return

        window_size = int(win_sec * fs)
        step_size = int(window_size * (1 - overlap_pct / 100))

        self.progress["maximum"] = len(files)
        self.progress["value"] = 0

        for file in files:
            self.log(f"處理：{file}")
            try:
                df = pd.read_csv(os.path.join(input_dir, file))
                time_column = None
                if df.columns[0].lower() in ["time", "timestamp"]:
                    time_column = df.columns[0]
                    time_data = df[time_column]
                    df = df.drop(columns=[time_column])
                n = len(df)
                bandpassed_data = {}
                relative_power_table = []

                for ch in df.columns:
                    x = df[ch].values
                    band_power_list = {b: [] for b in bands}
                    for start in range(0, n - window_size + 1, step_size):
                        segment = x[start:start + window_size]
                        freqs, psd = welch(segment, fs=fs, nperseg=window_size)
                        for band, (lo, hi) in bands.items():
                            band_power_list[band].append(band_power(freqs, psd, (lo, hi)))

                    avg_power = {band: np.mean(band_power_list[band]) for band in bands}
                    total_power = sum(avg_power.values())
                    # === 修改 analyze() 中的 rel_power 計算邏輯
                    if self.use_percentage.get():
                        rel_power = {band: (p / total_power * 100) if total_power > 0 else 0 for band, p in avg_power.items()}
                    else:
                        rel_power = {band: (p / total_power) if total_power > 0 else 0 for band, p in avg_power.items()}

                    rel_power["Channel"] = ch
                    relative_power_table.append(rel_power)

                    for band, (lo, hi) in bands.items():
                        filtered = bandpass_filter(x, lo, hi, fs)
                        bandpassed_data[f"{ch}_{band}"] = filtered

                # 輸出
                base = os.path.splitext(file)[0]
                rel_df = pd.DataFrame(relative_power_table)
                rel_df = pd.DataFrame(relative_power_table)
                rel_df = rel_df[["Channel"] + list(bands.keys())]
                
                # 攤平成單列格式（Fp1__Delta, Fp1__Theta, ...）
                flattened = {}
                for _, row in rel_df.iterrows():
                    for band in bands:
                        flattened[f"{row['Channel']}_{band}"] = row[band]
                flat_df = pd.DataFrame([flattened])
                flat_df.to_csv(os.path.join(output_dir, f"{base}_relative_band_power.csv"), index=False)


                bp_df = pd.DataFrame(bandpassed_data)
                if time_column:
                    bp_df.insert(0, time_column, time_data)
                bp_df.to_csv(os.path.join(output_dir, f"{base}_bandpassed_eeg.csv"), index=False)

                self.log("✅ 完成：" + file)
            except Exception as e:
                self.log(f"❌ 錯誤：{file}：{str(e)}")
            self.progress["value"] += 1

        self.log("🎉 所有檔案處理完畢！")

# === 主程式 ===
if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalyzerGUI(root)
    root.mainloop()
