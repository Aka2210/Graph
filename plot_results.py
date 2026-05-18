#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===== 全域字體設定：只放大字體，其餘大小不變 =====
plt.rcParams.update({
    "font.size": 25,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 22,
    "lines.linewidth": 2.4,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ===== 只要改這幾個參數即可 =====
X_START = 50
X_END = 450
X_STEP = 100

# OffPA 至少比其他三個方法大幾倍，才使用 broken y-axis
BREAK_RATIO = 5.0

# ===== 圖片輸出設定 =====
FIG_WIDTH = 8.8
FIG_HEIGHT = 6.8
PNG_DPI = 600


STYLES = {
    "DMTS": {
        "color": "blue",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 6
    },
    "TSMTA": {
        "color": "red",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.6,
        "markersize": 6
    },
    "SSSP": {
        "color": "green",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.6,
        "markersize": 6
    },
    "OffPA": {
        "color": "orange",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 2.6,
        "markersize": 6
    }
}


def get_x_config(x_type):
    if x_type == "sats":
        return {
            "x_col": "sat_num",
            "x_label": "Number of Satellites"
        }

    if x_type == "dests":
        return {
            "x_col": "dest_num",
            "x_label": "Number of Destinations"
        }

    raise ValueError("x_type must be either 'sats' or 'dests'.")


def collect_plot_data(df, metric, x_col, x_start, x_end, x_step):
    low_algos = ["DMTS", "TSMTA", "SSSP"]
    high_algos = ["OffPA"]
    algos = low_algos + high_algos

    std_col = f"{metric}_Std"
    selected_points = list(range(x_start, x_end + 1, x_step))

    plot_data = {}

    for algo in algos:
        df_algo = df[df["algo"] == algo].sort_values(x_col)
        df_algo = df_algo[df_algo[x_col].isin(selected_points)]

        if df_algo.empty:
            print(f"⚠ Warning: No data for {algo} in {metric}")
            continue

        x = df_algo[x_col]
        y = df_algo[metric] / 1000.0

        if std_col in df_algo.columns:
            y_err = df_algo[std_col] / 1000.0
        else:
            print(f"⚠ Warning: No std column '{std_col}' found. Plotting without error bars.")
            y_err = None

        plot_data[algo] = {
            "x": x,
            "y": y,
            "y_err": y_err
        }

    return plot_data


def should_use_broken_axis(plot_data, break_ratio):
    """
    若 OffPA 明顯比 DMTS / TSMTA / SSSP 大很多，才使用 broken y-axis。
    判斷條件：
        min(OffPA) / max(DMTS, TSMTA, SSSP) >= break_ratio
    """
    low_algos = ["DMTS", "TSMTA", "SSSP"]

    low_values = []
    high_values = []

    for algo in low_algos:
        if algo in plot_data:
            low_values.extend(list(plot_data[algo]["y"]))

    if "OffPA" in plot_data:
        high_values.extend(list(plot_data["OffPA"]["y"]))

    if not low_values or not high_values:
        return False

    low_max = max(low_values)
    high_min = min(high_values)

    if low_max <= 0:
        return False

    ratio = high_min / low_max

    print(f"🔎 OffPA separation ratio = {ratio:.2f}")

    return ratio >= break_ratio


def add_top_legend(fig, axes):
    """
    合併所有 axes 的 legend，放在整張圖上方。
    """
    handles = []
    labels = []

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        ncol=4,
        frameon=True,
        fontsize=19,
        markerscale=1.5,
        handlelength=2.5,
        handletextpad=0.7,
        columnspacing=1.1,
        borderpad=0.45,
        labelspacing=0.4
    )


def save_figure(output_path):
    plt.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)

    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    print(f"📊 Saved PNG: {output_path}")
    print(f"📄 Saved PDF: {pdf_path}")

    plt.close()


def plot_metric_normal(
    plot_data,
    metric,
    output_path,
    x_start,
    x_end,
    x_step,
    x_label
):
    """
    一般 y-axis 圖。
    適合 RC / CC 這種 OffPA 沒有大到需要斷軸的情況。
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    algos = ["DMTS", "TSMTA", "SSSP", "OffPA"]

    for algo in algos:
        if algo not in plot_data:
            continue

        data = plot_data[algo]

        ax.errorbar(
            data["x"],
            data["y"],
            yerr=data["y_err"],
            label=algo,
            capsize=4,
            elinewidth=1.3,
            markeredgewidth=0.9,
            **STYLES[algo]
        )

    margin = x_step * 0.15

    ax.set_xlim(x_start - margin, x_end + margin)
    ax.set_xticks(list(range(x_start, x_end + 1, x_step)))

    ax.set_xlabel(x_label, labelpad=6)
    ax.set_ylabel(f"{metric} (K)", labelpad=6)

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=25)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    add_top_legend(fig, [ax])

    plt.tight_layout(rect=(0.02, 0.02, 1, 0.93), pad=0.6)

    save_figure(output_path)


def plot_metric_broken(
    plot_data,
    metric,
    output_path,
    x_start,
    x_end,
    x_step,
    x_label
):
    """
    Broken y-axis version:
    - ax_bottom: only DMTS / TSMTA / SSSP
    - ax_top: only OffPA
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={
            "height_ratios": [1, 3],
            "hspace": 0.08
        }
    )

    low_algos = ["DMTS", "TSMTA", "SSSP"]

    # 下方：低值算法
    for algo in low_algos:
        if algo not in plot_data:
            continue

        data = plot_data[algo]

        ax_bottom.errorbar(
            data["x"],
            data["y"],
            yerr=data["y_err"],
            label=algo,
            capsize=4,
            elinewidth=1.3,
            markeredgewidth=0.9,
            **STYLES[algo]
        )

    # 上方：OffPA
    if "OffPA" in plot_data:
        data = plot_data["OffPA"]

        ax_top.errorbar(
            data["x"],
            data["y"],
            yerr=data["y_err"],
            label="OffPA",
            capsize=4,
            elinewidth=1.3,
            markeredgewidth=0.9,
            **STYLES["OffPA"]
        )

    # ===== 自動設定 y-axis 範圍 =====
    low_values = []
    high_values = []

    for algo in low_algos:
        if algo in plot_data:
            low_values.extend(list(plot_data[algo]["y"]))

    if "OffPA" in plot_data:
        high_values.extend(list(plot_data["OffPA"]["y"]))

    if not low_values:
        raise ValueError(f"No low-value algorithm data found for metric {metric}.")
    if not high_values:
        raise ValueError(f"No OffPA data found for metric {metric}.")

    low_min = min(low_values)
    low_max = max(low_values)
    high_min = min(high_values)
    high_max = max(high_values)

    low_range = low_max - low_min
    high_range = high_max - high_min

    if low_range == 0:
        low_range = max(abs(low_max), 1) * 0.1

    if high_range == 0:
        high_range = max(abs(high_max), 1) * 0.1

    bottom_lower = max(0, low_min - 0.20 * low_range)
    bottom_upper = low_max + 0.25 * low_range
    ax_bottom.set_ylim(bottom_lower, bottom_upper)

    top_lower = max(
        high_min - 0.05 * high_range,
        high_min * 0.80,
        0
    )
    top_upper = high_max + 0.10 * high_range
    ax_top.set_ylim(top_lower, top_upper)

    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
    ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # ===== x-axis 設定 =====
    margin = x_step * 0.15
    ax_bottom.set_xlim(x_start - margin, x_end + margin)
    ax_bottom.set_xticks(list(range(x_start, x_end + 1, x_step)))

    ax_bottom.set_xlabel(x_label, labelpad=6)

    # 共用 y label
    fig.text(
        0.015, 0.5,
        f"{metric} (K)",
        va="center",
        rotation="vertical",
        fontsize=30
    )

    for ax in [ax_top, ax_bottom]:
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=25)

    # 隱藏中間 spine
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    # ===== 畫 y-axis break 的斜線 =====
    d = 0.012

    kwargs = dict(
        transform=ax_top.transAxes,
        color="k",
        clip_on=False,
        linewidth=1.2
    )

    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)

    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    add_top_legend(fig, [ax_bottom, ax_top])

    plt.tight_layout(rect=(0.06, 0.02, 1, 0.93), pad=0.6)

    save_figure(output_path)


def plot_metric_auto(
    df,
    metric,
    output_path,
    x_start,
    x_end,
    x_step,
    x_col,
    x_label,
    break_ratio
):
    plot_data = collect_plot_data(
        df=df,
        metric=metric,
        x_col=x_col,
        x_start=x_start,
        x_end=x_end,
        x_step=x_step
    )

    use_broken = should_use_broken_axis(plot_data, break_ratio)

    if use_broken:
        print(f"✂ Using broken y-axis for {metric}")
        plot_metric_broken(
            plot_data=plot_data,
            metric=metric,
            output_path=output_path,
            x_start=x_start,
            x_end=x_end,
            x_step=x_step,
            x_label=x_label
        )
    else:
        print(f"📈 Using normal y-axis for {metric}")
        plot_metric_normal(
            plot_data=plot_data,
            metric=metric,
            output_path=output_path,
            x_start=x_start,
            x_end=x_end,
            x_step=x_step,
            x_label=x_label
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot simulation results with automatic broken y-axis."
    )

    parser.add_argument(
        "excel_path",
        help="Path to the results Excel file."
    )

    parser.add_argument(
        "--x",
        choices=["sats", "dests"],
        required=True,
        help="Choose x-axis type: 'sats' for number of satellites, 'dests' for number of destinations."
    )

    parser.add_argument(
        "--x-start",
        type=int,
        default=X_START,
        help="Start value of x-axis."
    )

    parser.add_argument(
        "--x-end",
        type=int,
        default=X_END,
        help="End value of x-axis."
    )

    parser.add_argument(
        "--x-step",
        type=int,
        default=X_STEP,
        help="Step size of x-axis."
    )

    parser.add_argument(
        "--break-ratio",
        type=float,
        default=BREAK_RATIO,
        help="Use broken y-axis only when min(OffPA) / max(other methods) >= this ratio."
    )

    args = parser.parse_args()

    excel_path = args.excel_path

    if not os.path.exists(excel_path):
        print(f"❌ File not found: {excel_path}")
        sys.exit(1)

    x_config = get_x_config(args.x)
    x_col = x_config["x_col"]
    x_label = x_config["x_label"]

    print(f"📘 Loading Excel: {excel_path}")
    print(f"📌 X-axis mode: {args.x}")
    print(f"📌 X-axis label: {x_label}")
    print(f"📌 Break ratio threshold: {args.break_ratio}")

    df = pd.read_excel(excel_path)

    # 從 graph 欄位解析 x 軸數量，例如 graph_50, graph_150, ...
    df[x_col] = df["graph"].str.extract(r"graph_(\d+)").astype(int)

    excel_name = os.path.splitext(os.path.basename(excel_path))[0]

    base_dir = f"img/{excel_name}_{args.x}"
    os.makedirs(base_dir, exist_ok=True)

    metrics = ["Total", "BC", "CC", "RC"]

    for metric in metrics:
        metric_dir = os.path.join(base_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)

        filename = f"{excel_name}_{args.x}_{metric}.png"
        output_path = os.path.join(metric_dir, filename)

        plot_metric_auto(
            df=df,
            metric=metric,
            output_path=output_path,
            x_start=args.x_start,
            x_end=args.x_end,
            x_step=args.x_step,
            x_col=x_col,
            x_label=x_label,
            break_ratio=args.break_ratio
        )

    print("🎉 All plots generated successfully!")


if __name__ == "__main__":
    main()