#!/usr/bin/env python3
"""算子契约探针模板（昇腾 NPU）。

用途：算子文档对 layout / head 映射 / 归一化 / 衰减语义描述不清时，
用**退化输入**把语义直接读出来，不要靠猜或反复调参。

使用步骤：
  1. 按目标算子改 `CALL()`：签名、关键字参数名、返回是 tuple 还是单张量。
  2. 按目标约束改 `CFG`（dtype、Dk/Dv、Nk/Nv、device）。
  3. 运行：python operator-contract-probe.py --mode layout|headmap|norm|all

判读：
  * layout : 候选布局 A / A^T 分别送入并与参考比对，max_abs 小的即真实布局。
  * headmap: 每个 value head 用到哪个 q/k head（repeat_interleave 还是 tile）。
  * norm   : 输入乘不同系数后输出是否同比放大，判断算子是否内部做 L2 norm。

注意：本文件是模板，不是通用驱动器；不同算子的入参语义不同，必须先改 CALL。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch_npu


@dataclass
class CFG:
    dtype: torch.dtype = torch.bfloat16
    nk: int = 2          # key heads
    nv: int = 6          # value heads
    dk: int = 128        # key dim
    dv: int = 128        # value dim
    scale: float | None = None  # None 表示 dk ** -0.5


CFG_ = CFG()


def CALL(q, k, v, beta, g, state, lengths, scale):
    """ADAPT ME：替换为目标算子的真实调用。

    约定输入（TND 布局，单 batch）：
      q/k [T, Nk, Dk]，v [T, Nv, Dv]，beta/g [T, Nv]，state [N, ...]，lengths [N] int32
    返回 (o, final_state)。
    """
    return torch_npu.npu_chunk_gated_delta_rule(
        q, k, v,
        beta=beta,
        initial_state=state,
        actual_seq_lengths=lengths,
        scale=scale,
        g=g,
    )


def dev():
    # 用 "npu:N" 而非整数索引，避免不同 torch 版本把 int 解释成别的语义
    return f"npu:{torch.npu.current_device()}"


def zeros(*shape, dtype=None):
    return torch.zeros(*shape, dtype=dtype or CFG_.dtype, device=dev())


def scale_of():
    return CFG_.scale if CFG_.scale is not None else CFG_.dk ** -0.5


def probe_layout():
    """T=1, beta=0, g=0 → out = f(S, q) * scale，用非对称 S 定 layout。"""
    torch.manual_seed(0)
    H, K, V = CFG_.nv, CFG_.dk, CFG_.dv
    S = torch.randn(H, K, V) * 0.5          # 参考语义：[H, K, V]
    q = torch.randn(1, CFG_.nk, CFG_.dk)
    k = torch.randn(1, CFG_.nk, CFG_.dk)
    beta = torch.zeros(1, CFG_.nv)
    g = torch.zeros(1, CFG_.nv)
    v = torch.zeros(1, CFG_.nv, CFG_.dv)
    sc = scale_of()

    # 参考：按 q 的 head 展开到 Nv（repeat_interleave 是最常见约定；如不同请改）
    def expand(t, target):
        return t if t.shape[0] == target else t.repeat_interleave(target // t.shape[0], dim=0)

    qr = expand(q[0].float(), H)
    ref = (torch.einsum("hkv,hk->hv", S, qr) * sc).cpu()

    cands = {
        "state[H,K,V]": S.unsqueeze(0),
        "state[H,V,K]": S.transpose(-1, -2).unsqueeze(0),
    }
    print("\n[layout] 每个候选布局与参考的误差（越小越可能正确）")
    for name, st in cands.items():
        o, _ = CALL(
            q.to(CFG_.dtype).npu(), k.to(CFG_.dtype).npu(), v.to(CFG_.dtype).npu(),
            beta.to(CFG_.dtype).npu(), g.npu(),
            st.to(CFG_.dtype).contiguous().npu(),
            torch.tensor([1], dtype=torch.int32).npu(), sc,
        )
        torch.npu.synchronize()
        d = (o.squeeze(0).float().cpu() - ref).abs()
        print(f"  {name:16} max_abs={d.max().item():.6f}  mean_abs={d.mean().item():.8f}")


def probe_headmap():
    """S 只在 (k=0,v=0) 有值，q 各 head 取不同常数 → 读出 head 映射。"""
    print("\n[headmap] value head → q head")
    H = CFG_.nv
    q = torch.zeros(1, CFG_.nk, CFG_.dk)
    for j in range(CFG_.nk):
        q[0, j, 0] = j + 1.0
    k = torch.zeros(1, CFG_.nk, CFG_.dk)
    v = torch.zeros(1, CFG_.nv, CFG_.dv)
    beta = torch.zeros(1, CFG_.nv)
    g = torch.zeros(1, CFG_.nv)

    # 两种候选 state 布局都试，取误差小者（与 probe_layout 结论交叉验证）
    for name, st in {
        "state[H,K,V]": torch.zeros(1, H, CFG_.dk, CFG_.dv),
        "state[H,V,K]": torch.zeros(1, H, CFG_.dv, CFG_.dk),
    }.items():
        st = st.clone()
        for h in range(H):
            # 把 S[h, 0, 0] 置 1（两种布局下都是 k=0, v=0 位置）
            st[0, h, 0, 0] = 1.0
        o, _ = CALL(
            q.to(CFG_.dtype).npu(), k.to(CFG_.dtype).npu(), v.to(CFG_.dtype).npu(),
            beta.to(CFG_.dtype).npu(), g.npu(), st.to(CFG_.dtype).npu(),
            torch.tensor([1], dtype=torch.int32).npu(), 1.0,
        )
        torch.npu.synchronize()
        vals = [round(x, 3) for x in o[0, :, 0].float().cpu().tolist()]
        print(f"  {name:16} 每 v head 读到的 q head 系数: {vals}")


def probe_norm():
    """q/k 乘不同系数：内部做 L2 norm 时输出不变，否则同比放大。"""
    print("\n[norm] 是否内部做 q/k L2 norm")
    torch.manual_seed(1)
    T = 8
    q = torch.randn(1, T, CFG_.nk, CFG_.dk)
    k = torch.randn(1, T, CFG_.nk, CFG_.dk)
    v = torch.randn(1, T, CFG_.nv, CFG_.dv)
    beta = torch.rand(1, T, CFG_.nv)
    g = -torch.rand(1, T, CFG_.nv)
    st = torch.zeros(1, CFG_.nv, CFG_.dv, CFG_.dk)
    lengths = torch.tensor([T], dtype=torch.int32)

    def run(qq, kk):
        o, _ = CALL(
            qq.to(CFG_.dtype).npu(), kk.to(CFG_.dtype).npu(), v.to(CFG_.dtype).npu(),
            beta.to(CFG_.dtype).npu(), g.npu(), st.to(CFG_.dtype).npu(), lengths.npu(),
            scale_of(),
        )
        torch.npu.synchronize()
        return o.float().cpu()

    base = run(q, k)
    scaled = run(q * 2.0, k * 3.0)
    ratio = (scaled.norm() / base.norm()).item()
    print(f"  q×2、k×3 后输出范数比 = {ratio:.4f}")
    print("  判读：≈1 → 内部做 L2 norm；≈6（2×3）→ 不做 norm，调用方需自行归一化")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["layout", "headmap", "norm", "all"])
    args = ap.parse_args()
    print("device:", torch.npu.get_device_name(dev()), "| dtype:", CFG_.dtype)
    if args.mode in ("layout", "all"):
        probe_layout()
    if args.mode in ("headmap", "all"):
        probe_headmap()
    if args.mode in ("norm", "all"):
        probe_norm()
