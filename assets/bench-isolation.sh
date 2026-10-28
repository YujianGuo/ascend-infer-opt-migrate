#!/usr/bin/env bash
# 隔离负载基准模板（vLLM 服务；其它框架按同样口径替换请求命令）。
#
# 目的：让"被测优化"对应的阶段主导指标，避免无关阶段把效应淹没：
#   prefill : 长输入 + 输出 1 token + 并发 1  → TTFT 由 prefill 主导
#   decode  : 短输入 + 较长输出 + 并发 N      → TPOT/ITL 由 decode 主导
#   mixed   : 线上混合负载                    → 只作辅证，不能单凭它判定
#
# 用法：
#   ./bench-isolation.sh --tag t1 --mode prefill \
#       --base-url http://127.0.0.1:8001 --model my-model --tokenizer /path/to/tok --reps 3
#
# 判读：同一会话内 reps 的极差是"重复噪声"；不同会话（不同 tag）之间的差是"会话漂移"。
#       效应量必须大于会话漂移才能下结论（见 references/validation.md 实验设计一节）。
set -euo pipefail

TAG=""; MODE="prefill"; BASE_URL="http://127.0.0.1:8001"; MODEL=""; TOKENIZER=""
REPS=3; OUT_ROOT="${OUT_ROOT:-./bench_isolated}"
INPUT_LEN=""; OUTPUT_LEN=""; CONC=""; PROMPTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --out) OUT_ROOT="$2"; shift 2;;
    --input-len) INPUT_LEN="$2"; shift 2;;
    --output-len) OUTPUT_LEN="$2"; shift 2;;
    --concurrency) CONC="$2"; shift 2;;
    --prompts) PROMPTS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$TAG" && -n "$MODEL" ]] || { echo "--tag and --model are required" >&2; exit 2; }

case "$MODE" in
  prefill) INPUT_LEN="${INPUT_LEN:-8192}"; OUTPUT_LEN="${OUTPUT_LEN:-1}";   CONC="${CONC:-1}"; PROMPTS="${PROMPTS:-8}";;
  decode)  INPUT_LEN="${INPUT_LEN:-1024}"; OUTPUT_LEN="${OUTPUT_LEN:-256}"; CONC="${CONC:-8}"; PROMPTS="${PROMPTS:-32}";;
  mixed)   INPUT_LEN="${INPUT_LEN:-8192}"; OUTPUT_LEN="${OUTPUT_LEN:-1024}";CONC="${CONC:-8}"; PROMPTS="${PROMPTS:-32}";;
  *) echo "--mode must be prefill|decode|mixed" >&2; exit 2;;
esac

HOSTPORT="${BASE_URL#http://}"; HOST="${HOSTPORT%%:*}"; PORT="${HOSTPORT##*:}"
RESULT_DIR="${OUT_ROOT}/${TAG}"; mkdir -p "$RESULT_DIR"

TOKENIZER_ARG=()
[[ -n "$TOKENIZER" ]] && TOKENIZER_ARG=(--tokenizer "$TOKENIZER")

echo "== [${TAG}/${MODE}] in=${INPUT_LEN} out=${OUTPUT_LEN} conc=${CONC} prompts=${PROMPTS} reps=${REPS}"

for i in $(seq 1 "$REPS"); do
  vllm bench serve \
    --backend vllm --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$PROMPTS" --max-concurrency "$CONC" --ignore-eos \
    --model "$MODEL" "${TOKENIZER_ARG[@]}" \
    --endpoint /v1/completions --request-rate inf --seed 1024 \
    --host "$HOST" --port "$PORT" \
    --save-result --result-dir "$RESULT_DIR" --result-filename "rep_$i.json" > /dev/null 2>&1
  python3 - "$RESULT_DIR/rep_$i.json" "$i" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  rep{sys.argv[2]}: ttft mean={r['mean_ttft_ms']:8.1f} med={r['median_ttft_ms']:8.1f} "
      f"p99={r['p99_ttft_ms']:8.1f} | tpot={r['mean_tpot_ms']:6.2f} | "
      f"out={r['output_throughput']:8.2f} tok/s | in={r['total_token_throughput']:8.1f} tok/s")
PY
done

python3 - "$RESULT_DIR" "$REPS" "$TAG" "$MODE" <<'PY'
import json, statistics as st, sys
d, n, tag, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
rows = [json.load(open(f"{d}/rep_{i}.json")) for i in range(1, n + 1)]
keys = [("mean_ttft_ms", "TTFT mean ms"), ("median_ttft_ms", "TTFT med ms"),
        ("p99_ttft_ms", "TTFT p99 ms"), ("mean_tpot_ms", "TPOT mean ms"),
        ("output_throughput", "output tok/s"), ("total_token_throughput", "input+output tok/s")]
print(f"== [{tag}/{mode}] 本会话汇总（极差=重复噪声）")
for k, label in keys:
    v = [float(r[k]) for r in rows]
    spread = (max(v) - min(v)) / st.mean(v) * 100 if st.mean(v) else 0.0
    print(f"{label:>18}: mean={st.mean(v):9.2f} min={min(v):9.2f} max={max(v):9.2f} 极差={spread:5.2f}%")
PY
echo "原始结果: $RESULT_DIR/rep_*.json"
