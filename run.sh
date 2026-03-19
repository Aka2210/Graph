set -euo pipefail
if [ $# -lt 2 ]; then
    echo "用法: ./run.sh <次數> <config.json>"
    exit 1
fi
N=$1
CONFIG=$2
BASE_NSATS=$(jq -r '.start_sats' "$CONFIG")
for ((i=1; i<=N; i++))
do
    NSATS=$((BASE_NSATS + 50 * (i - 1)))
    TMP_CONFIG=$(mktemp /tmp/config.XXXXXX.json)
    jq --argjson nsats "$NSATS" --argjson seed "$i" \
       '.n_sats=$nsats | .seed_offset=$seed' \
       "$CONFIG" > "$TMP_CONFIG"
    echo "=== 第 $i 次 === n_sats=$NSATS  config=$TMP_CONFIG"
    python Random_Orbit.py "$TMP_CONFIG"
    python main.py "$TMP_CONFIG"
    rm -f "$TMP_CONFIG"
done
