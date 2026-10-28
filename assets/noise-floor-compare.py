#!/usr/bin/env python3
"""噪声底 vs 跨配置差异比较模板。

用途：非确定模型（MoE 路由、投机解码、bf16 归约）在判断"优化是否造成精度回归/性能变化"前，
必须先测**同配置重复噪声**；跨配置差异小于噪声底时不得下结论。

输入：若干个 JSON 文件，每个文件是 {指标名: 数值} 的扁平字典，例如
  same_1.json / same_2.json / same_3.json   # 同一配置重复
  base.json                                  # baseline（T0）
  cand.json                                  # candidate（T1，可多个）

用法：
  python noise-floor-compare.py --same same_1.json same_2.json same_3.json \
      --baseline base.json --candidate cand_1.json cand_2.json

判读：
  * spread  : 同配置重复的极差（噪声底）
  * delta   : candidate 均值与 baseline 的差
  * 结论     : |delta| <= spread → "噪声内"，否则 "超出噪声"
"""

from __future__ import annotations

import argparse
import json
import statistics as st


def load(path: str) -> dict[str, float]:
    with open(path) as f:
        d = json.load(f)
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same", nargs="+", required=True, help="同配置重复的 JSON")
    ap.add_argument("--baseline", required=True, help="基线 JSON（T0）")
    ap.add_argument("--candidate", nargs="+", required=True, help="候选 JSON（T1，可多个）")
    ap.add_argument("--lower-better", nargs="*", default=[], help="越小越好的指标名（如 ttft_ms、tpot_ms）")
    args = ap.parse_args()

    same = [load(p) for p in args.same]
    base = load(args.baseline)
    cand = [load(p) for p in args.candidate]

    keys = sorted(set(base) & set().union(*same, *cand))
    print(f"{'metric':28} {'noise(spread)':>14} {'base':>10} {'cand_mean':>10} {'delta':>10} {'verdict':>12}")
    for k in keys:
        sv = [d[k] for d in same if k in d]
        cv = [d[k] for d in cand if k in d]
        if not sv or not cv or k not in base:
            continue
        spread = max(sv) - min(sv)
        cm = st.mean(cv)
        delta = cm - base[k]
        verdict = "噪声内" if abs(delta) <= spread else ("超出噪声↓" if delta < 0 else "超出噪声↑")
        if k in args.lower_better:
            verdict += " (越低越好)"
        print(f"{k:28} {spread:14.6f} {base[k]:10.6f} {cm:10.6f} {delta:+10.6f} {verdict:>12}")

    print("\n提示：spread 只反映**同一会话内**的重复噪声；跨会话还要用不同会话的同配置数据"
          "\n估计会话漂移，效应需大于两者中较大的那个才可下结论。")


if __name__ == "__main__":
    main()
