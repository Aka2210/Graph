#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric, output_path):
    """
    畫出某一 cost metric 的三條曲線
    """
    plt.figure(figsize=(10, 6))

    for algo in ["DMTS", "OffPA", "TSMTA", "SSSP"]:
        df_algo = df[df["algo"] == algo].sort_values("sat_num")

        if df_algo.empty:
            print(f"⚠ Warning: No data for {algo} in {metric}")
            continue

        plt.plot(df_algo["sat_num"], df_algo[metric],
         marker='o', label=algo)

    plt.xlabel("Number of Satellites")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison Among Algorithms")
    plt.legend()
    plt.grid(True)
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

    df["sat_num"] = df["graph"].str.extract(r"graph_(\d+)").astype(int)

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

        # PNG 檔名： <excel_name>_<metric>.png
        filename = f"{excel_name}_{metric}.png"

        output_path = os.path.join(metric_dir, filename)

        plot_metric(df, metric, output_path)

    print("🎉 All plots generated successfully!")


if __name__ == "__main__":
    main()
