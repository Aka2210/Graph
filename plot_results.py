#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== 全域字體設定：適合論文圖 =====
plt.rcParams.update({
    "font.size": 18,         # 基本字體
    "axes.titlesize": 22,    # 圖標題
    "axes.labelsize": 20,    # x/y 軸標籤
    "xtick.labelsize": 18,   # x 軸刻度字
    "ytick.labelsize": 18,   # y 軸刻度字
    "legend.fontsize": 18    # 圖例字體
})

def plot_metric(df, metric, output_path):
    """
    畫出某一 cost metric 的四條曲線 (包含誤差棒)
    """
    plt.figure(figsize=(12, 8))

    algos = ["DMTS", "TSMTA", "SSSP", "OffPA"]

    styles = {
        "DMTS":  {"color": "blue",   "marker": "o", "linestyle": "-",  "linewidth": 2.5, "markersize": 9},
        "TSMTA": {"color": "red",    "marker": "s", "linestyle": "--", "linewidth": 2.5, "markersize": 9},
        "SSSP":  {"color": "green",  "marker": "^", "linestyle": "-.", "linewidth": 2.5, "markersize": 9},
        "OffPA": {"color": "orange", "marker": "D", "linestyle": ":",  "linewidth": 2.5, "markersize": 9}
    }

    std_col = f"{metric}_Std"

    for algo in algos:
        df_algo = df[df["algo"] == algo].sort_values("sat_num")

        if df_algo.empty:
            print(f"⚠ Warning: No data for {algo} in {metric}")
            continue

        x = df_algo["sat_num"]
        y = df_algo[metric]

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
            capsize=6,
            elinewidth=1.8,
            markeredgewidth=1.2,
            **style
        )

    plt.xlabel("Number of Satellites", labelpad=10)
    plt.ylabel(metric, labelpad=10)
    plt.title(f"{metric} Comparison", pad=15)

    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 刻度字體再明確放大
    plt.xticks(df["sat_num"].sort_values().unique())
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

    # 從 graph 欄位解析衛星數量 (假設格式為 graph_60_avg10)
    df["sat_num"] = df["graph"].str.extract(r"graph_(\d+)").astype(int)

    excel_name = os.path.splitext(os.path.basename(excel_path))[0]

    base_dir = f"img/{excel_name}"
    os.makedirs(base_dir, exist_ok=True)

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