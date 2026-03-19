#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(df, metric, output_path):
    """
    畫出某一 cost metric 的三條曲線 (包含誤差棒)
    """
    plt.figure(figsize=(10, 6))
    
    # 這裡的順序決定圖例順序
    algos = ["DMTS", "TSMTA", "SSSP", "OffPA"] 
    
    # 定義每個演算法的樣式，讓視覺更一致
    styles = {
        "DMTS":  {"color": "blue",   "marker": "o", "linestyle": "-"},
        "TSMTA": {"color": "red",    "marker": "s", "linestyle": "--"},
        "SSSP":  {"color": "green",  "marker": "^", "linestyle": "-."},
        "OffPA": {"color": "orange", "marker": "D", "linestyle": ":"}
    }

    # 對應的標準差欄位名稱
    std_col = f"{metric}_Std"

    for algo in algos:
        # [修改點 A] sort_values 改為 "n_dests"
        df_algo = df[df["algo"] == algo].sort_values("dest_num")

        if df_algo.empty:
            print(f"⚠ Warning: No data for {algo} in {metric}")
            continue

        # [修改點 B] X 軸數據改為 "n_dests"
        x = df_algo["n_dests"]
        y = df_algo[metric]
        
        # 檢查是否存在對應的標準差欄位
        if std_col in df_algo.columns:
            y_err = df_algo[std_col]
        else:
            print(f"⚠ Warning: No std column '{std_col}' found. Plotting without error bars.")
            y_err = None

        style = styles.get(algo, {})
        
        plt.errorbar(
            x, y, 
            yerr=y_err, 
            label=algo,
            capsize=5,       
            elinewidth=1.5,
            markeredgewidth=1,
            **style
        )

    # [修改點 C] Label 改為 Destinations
    plt.xlabel("Number of Destinations")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison (by Destinations)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"📊 Saved: {output_path}")

    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results.xlsx>")
        sys.exit(1)

    excel_path = sys.argv[1]

    if not os.path.exists(excel_path):
        print(f"❌ File not found: {excel_path}")
        sys.exit(1)

    print(f"📘 Loading Excel: {excel_path}")
    df = pd.read_excel(excel_path)

    # [修改點 D] 移除原本的 sat_num 解析
    # df["sat_num"] = df["graph"].str.extract(r"graph_(\d+)").astype(int)
    df["dest_num"] = df["graph"].str.extract(r"graph_(\d+)").astype(int)
    
    # 確保 Excel 裡面有 n_dests 欄位 (這是你參數檔裡的變數名稱)
    if "n_dests" not in df.columns:
        print("❌ 錯誤：Excel 中找不到 'n_dests' 欄位。")
        print("請確認 main.py 是否有將 'n_dests' 參數寫入 Excel。")
        sys.exit(1)

    # Excel 名稱
    excel_name = os.path.splitext(os.path.basename(excel_path))[0]

    # 輸出資料夾
    base_dir = f"img/{excel_name}"
    os.makedirs(base_dir, exist_ok=True)

    # 四種 metric
    metrics = ["Total", "BC", "CC", "RC"]

    for metric in metrics:
        metric_dir = os.path.join(base_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)

        filename = f"{excel_name}_{metric}.png"
        output_path = os.path.join(metric_dir, filename)

        plot_metric(df, metric, output_path)

    print("🎉 All plots generated successfully!")


if __name__ == "__main__":
    main()