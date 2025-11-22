#!/bin/bash

if [ $# -lt 2 ]; then
    echo "用法: ./run.sh <次數> <config.json>"
    exit 1
fi

N=$1
CONFIG=$2

for ((i=1; i<=N; i++))
do
    echo "=== 第 $i 次 ==="

    python Random_Orbit.py "$CONFIG"
    python main.py "$CONFIG"

done

python plot_results.py "$CONFIG"

# x = num_sats, y = total cost
# ATDMST+offpa+dmst+spt+tsmta 圖表
