set -euo pipefail
if [ $# -lt 2 ]; then
    echo "用法: ./run_dests.sh <次數> <config.json>"
    echo "範例: ./run_dests.sh 5 config.json (將執行 dests=10, 20, 30, 40, 50)"
    exit 1
fi
N=$1
CONFIG=$2
if ! command -v jq &> /dev/null; then
    echo "錯誤: 未安裝 jq，請先執行 sudo apt install jq"
    exit 1
fi
for ((i=1; i<=N; i++))
do
    NDESTS=$((50 * i))
    TMP_CONFIG=$(mktemp /tmp/config_dests.XXXXXX.json)
    jq --argjson ndests "$NDESTS" --argjson seed "$i" \
        '.n_dests=$ndests | .seed_offset=$seed' \
        "$CONFIG" > "$TMP_CONFIG"
    echo "=== 第 $i 次 === n_dests=$NDESTS  config=$TMP_CONFIG"
    python Random_Orbit.py "$TMP_CONFIG" 
    python main_dest.py "$TMP_CONFIG"
    rm -f "$TMP_CONFIG"
done
echo "所有實驗執行完畢！"