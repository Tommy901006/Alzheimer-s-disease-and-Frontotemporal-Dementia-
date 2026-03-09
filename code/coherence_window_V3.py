import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from scipy import signal
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage


class CoherenceAnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("EEG Coherence Calculator")
        master.geometry("950x700")

        self.recipient_email = None
        self.sender_email = None
        self.sender_password = None

        self.build_interface()

    def build_interface(self):
        # ===== Folder and column selection =====
        frame_path = ttk.LabelFrame(self.master, text="Folder and Column Selection", padding=10)
        frame_path.pack(fill='x', padx=10, pady=5)

        ttk.Button(frame_path, text="Select Folder", command=self.select_folder).pack(anchor='w')

        self.lbl_folder = ttk.Label(frame_path, text="")
        self.lbl_folder.pack(anchor='w', pady=5)

        self.combo_cols = []
        frame_cols = ttk.Frame(frame_path)
        frame_cols.pack(anchor='w')

        for i in range(2):
            ttk.Label(frame_cols, text=f"Column {i+1}:").grid(row=i, column=0, sticky='e', padx=5, pady=2)
            cb = ttk.Combobox(frame_cols, width=35, state="readonly")
            cb.grid(row=i, column=1, padx=5, pady=2)
            self.combo_cols.append(cb)

        # ===== Settings =====
        frame_settings = ttk.LabelFrame(self.master, text="Settings", padding=10)
        frame_settings.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame_settings, text="Sampling Rate (Hz):").grid(row=0, column=0, sticky='w')
        self.entry_fs = ttk.Entry(frame_settings, width=10)
        self.entry_fs.insert(0, "1000")
        self.entry_fs.grid(row=0, column=1, padx=5, pady=2, sticky='w')

        self.var_window = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_settings, text="Enable Sliding Window", variable=self.var_window).grid(
            row=0, column=2, padx=10, pady=2, sticky='w'
        )

        self.var_per_segment = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_settings, text="Export Per Segment", variable=self.var_per_segment).grid(
            row=0, column=3, padx=10, pady=2, sticky='w'
        )

        self.var_plot = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_settings, text="Save Segment Trend Plots", variable=self.var_plot).grid(
            row=0, column=4, padx=10, pady=2, sticky='w'
        )

        self.var_email = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_settings, text="Email Result", variable=self.var_email).grid(
            row=0, column=5, padx=10, pady=2, sticky='w'
        )

        ttk.Label(frame_settings, text="Window Size:").grid(row=1, column=0, sticky='w')
        self.entry_window = ttk.Entry(frame_settings, width=10)
        self.entry_window.insert(0, "1000")
        self.entry_window.grid(row=1, column=1, padx=5, pady=2, sticky='w')

        ttk.Label(frame_settings, text="Overlap (%):").grid(row=1, column=2, sticky='w')
        self.entry_overlap = ttk.Entry(frame_settings, width=10)
        self.entry_overlap.insert(0, "50")
        self.entry_overlap.grid(row=1, column=3, padx=5, pady=2, sticky='w')

        ttk.Label(frame_settings, text="nperseg (blank = auto):").grid(row=1, column=4, sticky='w')
        self.entry_nperseg = ttk.Entry(frame_settings, width=10)
        self.entry_nperseg.insert(0, "")
        self.entry_nperseg.grid(row=1, column=5, padx=5, pady=2, sticky='w')

        # ===== Log =====
        frame_log = ttk.LabelFrame(self.master, text="Execution Log", padding=10)
        frame_log.pack(fill='both', expand=True, padx=10, pady=5)

        self.log = scrolledtext.ScrolledText(frame_log, height=15)
        self.log.pack(fill='both', expand=True)

        # ===== Start button =====
        ttk.Button(self.master, text="Start Batch Processing", command=self.start_processing).pack(pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        self.lbl_folder.config(text=folder)

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
        if not files:
            messagebox.showwarning("No Files", "No Excel or CSV files found in the selected folder.")
            return

        first_file = os.path.join(folder, files[0])

        try:
            df = self.read_table(first_file)
            cols = df.columns.tolist()
            for cb in self.combo_cols:
                cb['values'] = [''] + cols
                cb.set('')
            self.log_message(f"Loaded sample file for column selection: {files[0]}")
        except Exception as e:
            messagebox.showerror("Read Error", f"Failed to read sample file:\n{e}")

    def read_table(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif ext == '.csv':
            # 先嘗試 utf-8，再退回 cp950 / latin1
            try:
                return pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(path, encoding='cp950')
                except UnicodeDecodeError:
                    return pd.read_csv(path, encoding='latin1')
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def log_message(self, msg):
        self.log.insert(tk.END, msg + '\n')
        self.log.yview(tk.END)
        self.master.update_idletasks()

    def calculate_coherence(self, X, Y, fs=1000, nperseg=None):
        f, Cxy = signal.coherence(X, Y, fs=fs, nperseg=nperseg)
        return f, Cxy, np.mean(Cxy)

    def band_coherence(self, f, Cxy, band):
        bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 100)
        }

        low, high = bands[band]
        mask = (f >= low) & (f < high)

        if np.any(mask):
            return np.mean(Cxy[mask])
        return np.nan

    def get_valid_inputs(self):
        folder = self.lbl_folder.cget("text").strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Folder Error", "Please select a valid folder first.")
            return None

        selected_cols = [cb.get() for cb in self.combo_cols if cb.get()]
        if len(selected_cols) != 2:
            messagebox.showerror("Column Error", "Please select exactly 2 columns.")
            return None

        try:
            fs = float(self.entry_fs.get())
            if fs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Sampling rate must be a positive number.")
            return None

        use_window = self.var_window.get()
        export_segment = self.var_per_segment.get()
        plot_segment = self.var_plot.get()
        send_email = self.var_email.get()

        window_size = None
        overlap_ratio = 0
        step = None

        if use_window:
            try:
                window_size = int(self.entry_window.get())
                overlap_ratio = float(self.entry_overlap.get()) / 100.0
            except ValueError:
                messagebox.showerror("Input Error", "Window size and overlap must be numeric.")
                return None

            if window_size <= 0:
                messagebox.showerror("Input Error", "Window size must be > 0.")
                return None

            if not (0 <= overlap_ratio < 1):
                messagebox.showerror("Input Error", "Overlap must be between 0 and 99.")
                return None

            step = int(window_size * (1 - overlap_ratio))
            if step <= 0:
                messagebox.showerror("Input Error", "Step size becomes 0. Please reduce overlap.")
                return None

        nperseg_text = self.entry_nperseg.get().strip()
        nperseg = None
        if nperseg_text != "":
            try:
                nperseg = int(nperseg_text)
                if nperseg <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Input Error", "nperseg must be a positive integer or blank.")
                return None

        return {
            "folder": folder,
            "selected_cols": selected_cols,
            "fs": fs,
            "use_window": use_window,
            "export_segment": export_segment,
            "plot_segment": plot_segment,
            "send_email": send_email,
            "window_size": window_size,
            "overlap_ratio": overlap_ratio,
            "step": step,
            "nperseg": nperseg
        }

    def start_processing(self):
        config = self.get_valid_inputs()
        if config is None:
            return

        folder = config["folder"]
        selected_cols = config["selected_cols"]
        fs = config["fs"]
        use_window = config["use_window"]
        export_segment = config["export_segment"]
        plot_segment = config["plot_segment"]
        send_email = config["send_email"]
        window_size = config["window_size"]
        step = config["step"]
        nperseg = config["nperseg"]

        if send_email:
            self.recipient_email = simpledialog.askstring("Recipient Email", "Enter recipient email:")
            if not self.recipient_email:
                messagebox.showwarning("Email Cancelled", "Recipient email was not entered. Email sending disabled.")
                send_email = False
            else:
                self.sender_email = simpledialog.askstring("Sender Gmail", "Enter sender Gmail address:")
                self.sender_password = simpledialog.askstring("App Password", "Enter Gmail App Password:", show='*')

                if not self.sender_email or not self.sender_password:
                    messagebox.showwarning("Email Cancelled", "Sender email or app password missing. Email sending disabled.")
                    send_email = False

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
        if not files:
            messagebox.showwarning("No Files", "No Excel or CSV files found in the selected folder.")
            return

        summary_results = []
        segment_results = {}

        plot_dir = os.path.join(folder, "Segment_Plots")
        if plot_segment and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.log_message("===== Start Batch Processing =====")

        for file in files:
            path = os.path.join(folder, file)

            try:
                df = self.read_table(path)

                # 檢查欄位存在
                if selected_cols[0] not in df.columns or selected_cols[1] not in df.columns:
                    self.log_message(f"Skipped {file}: selected columns not found.")
                    continue

                # 兩欄一起 dropna，避免錯位
                pair_df = df[[selected_cols[0], selected_cols[1]]].dropna().reset_index(drop=True)

                if pair_df.empty:
                    self.log_message(f"Skipped {file}: no valid paired data after dropna.")
                    continue

                # 檢查數值型態
                x_series = pd.to_numeric(pair_df[selected_cols[0]], errors='coerce')
                y_series = pd.to_numeric(pair_df[selected_cols[1]], errors='coerce')
                valid_df = pd.DataFrame({
                    selected_cols[0]: x_series,
                    selected_cols[1]: y_series
                }).dropna().reset_index(drop=True)

                if valid_df.empty:
                    self.log_message(f"Skipped {file}: selected columns contain no valid numeric data.")
                    continue

                X = valid_df[selected_cols[0]].to_numpy()
                Y = valid_df[selected_cols[1]].to_numpy()
                length = len(valid_df)

                if use_window:
                    if length < window_size:
                        self.log_message(
                            f"Skipped {file}: data length ({length}) < window size ({window_size})."
                        )
                        continue

                    segs = []
                    coh_values = []

                    num_segments = (length - window_size) // step + 1

                    for i in range(num_segments):
                        s = i * step
                        e = s + window_size

                        if e > length:
                            break

                        seg_x = X[s:e]
                        seg_y = Y[s:e]

                        current_nperseg = nperseg if nperseg is not None else min(256, len(seg_x))
                        current_nperseg = min(current_nperseg, len(seg_x))

                        f, Cxy, avg = self.calculate_coherence(seg_x, seg_y, fs=fs, nperseg=current_nperseg)

                        band_results = {
                            band: self.band_coherence(f, Cxy, band)
                            for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
                        }

                        seg_info = {
                            "Segment": f"Segment{i+1}",
                            "File": file,
                            "Start_Index": s,
                            "End_Index": e - 1,
                            "Mean Coherence": avg,
                            **band_results
                        }

                        segs.append(seg_info)
                        coh_values.append(avg)

                    if not segs:
                        self.log_message(f"Skipped {file}: no valid segments generated.")
                        continue

                    band_means = {}
                    for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                        band_values = [seg[band] for seg in segs if not np.isnan(seg[band])]
                        band_means[band] = np.mean(band_values) if band_values else np.nan

                    summary_results.append({
                        "File": file,
                        "Mean Coherence": np.mean(coh_values) if coh_values else np.nan,
                        "Num Segments": len(segs),
                        **band_means
                    })

                    if export_segment:
                        segment_results[file] = segs

                    if plot_segment:
                        plt.figure(figsize=(10, 6))
                        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                            values = [seg[band] for seg in segs]
                            plt.plot(range(1, len(values) + 1), values, marker='o', label=band)

                        plt.title(f"{file} - Segment Coherence by Band")
                        plt.xlabel("Segment")
                        plt.ylabel("Coherence")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()

                        plot_path = os.path.join(plot_dir, f"{os.path.splitext(file)[0]}_segment_trend.png")
                        plt.savefig(plot_path, dpi=300)
                        plt.close()

                else:
                    current_nperseg = nperseg if nperseg is not None else min(256, len(X))
                    current_nperseg = min(current_nperseg, len(X))

                    f, Cxy, avg = self.calculate_coherence(X, Y, fs=fs, nperseg=current_nperseg)

                    band_results = {
                        band: self.band_coherence(f, Cxy, band)
                        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
                    }

                    summary_results.append({
                        "File": file,
                        "Mean Coherence": avg,
                        "Num Segments": 1,
                        **band_results
                    })

                self.log_message(f"Processed: {file}")

            except Exception as e:
                self.log_message(f"Error in {file}: {e}")

        if not summary_results:
            messagebox.showwarning("No Results", "No files were successfully processed.")
            self.log_message("No valid results generated.")
            return

        # ===== Save summary =====
        summary_path = os.path.join(folder, "EEG_Coherence_Summary.xlsx")
        pd.DataFrame(summary_results).to_excel(summary_path, index=False)
        self.log_message(f"Summary saved: {summary_path}")

        # ===== Save segment results =====
        seg_path = None
        if export_segment and segment_results:
            seg_path = os.path.join(folder, "EEG_Coherence_PerSegment.xlsx")
            with pd.ExcelWriter(seg_path, engine='openpyxl') as writer:
                for file, rows in segment_results.items():
                    df_seg = pd.DataFrame(rows)
                    sheet_name = os.path.splitext(file)[0][:31]
                    df_seg.to_excel(writer, sheet_name=sheet_name, index=False)

            self.log_message(f"Per-segment result saved: {seg_path}")

        # ===== Email =====
        if send_email and self.recipient_email and self.sender_email and self.sender_password:
            try:
                files_to_send = [summary_path]
                if seg_path is not None:
                    files_to_send.append(seg_path)

                self.send_email(
                    sender_email=self.sender_email,
                    sender_password=self.sender_password,
                    to_email=self.recipient_email,
                    attachments=files_to_send
                )
                self.log_message("✅ Email sent successfully.")
            except Exception as e:
                self.log_message(f"❌ Email failed: {e}")

        self.log_message("===== Batch Processing Finished =====")
        messagebox.showinfo("Done", "All files processed successfully.")

    def send_email(self, sender_email, sender_password, to_email, attachments):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        msg = EmailMessage()
        msg['Subject'] = "EEG Coherence Analysis Report"
        msg['From'] = sender_email
        msg['To'] = to_email
        msg.set_content("Attached are the EEG coherence analysis results.")

        for path in attachments:
            with open(path, 'rb') as f:
                data = f.read()
                name = os.path.basename(path)
                msg.add_attachment(
                    data,
                    maintype='application',
                    subtype='octet-stream',
                    filename=name
                )

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = CoherenceAnalysisGUI(root)
    root.mainloop()