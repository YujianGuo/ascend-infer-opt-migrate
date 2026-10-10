# 模型轴：迁移计算语义与运行契约

用于 M 与 HM。在目标模型本身的参考实现上验证，不要求源模型和目标模型生成相同文本。

## 结构映射

从模型 config、forward、权重加载和实际 tensor 取得下表信息；不要只依赖模型名、module 名或一张旧结构图。

| 模型契约 | 对照内容 | 典型迁移风险 |
|---|---|---|
| Attention | MHA/GQA/MQA/MLA、滑窗/稀疏、混合 full/linear attention、层类别 | 把压缩 latent KV 当完整 K/V，混淆线性状态与 attention cache |
| 维度 | hidden/head_dim、Q/KV heads、层数、shard 形状、packed/变长输入 | TP heads 切分不成立，固定维度 kernel 越界 |
| RoPE/Norm | 普通 RoPE/MRoPE、旋转范围/顺序、位置输入、scale、epsilon、RMS weight offset、残差语义 | 输出形状相同但数学含义不同 |
| FFN/MoE | 激活、gate/up/down 顺序、专家数/top-k、路由权重、共享专家 | 门控/专家顺序、路由和 combine 语义改变 |
| 权重/量化 | 名称映射、QKV packing、转置、weight loader、post-load、K/V scale 和 offset | 元数据在 stacked 参数加载中合并或丢失 |
| 缓存 | cache 类型、每层结构、layout/block、slot、position、量化元数据、状态更新 | 仅替换读取算子，遗漏写入/分配/图 replay |
| 运行接口 | forward metadata、输出消费者、hook、多分支、in-place/alias、图模式 | 融合移除仍被外部消费的中间值 |
| 多模态 | vision/语言边界、processor、视觉 token 与 position、变长拼接 | 沿用语言模型 shape 与位置假设 |

## 可复用性判断

对每个源优化给出目标子图和语义条件：满足时复用；不同但可建立等价映射时适配；没有对应结构时标记不适用。例如 MoE dispatch 不能迁入没有专家结构的 Dense FFN；MLA 压缩缓存不能仅改名字成为 GQA cache。

固定维度融合迁移前检查目标 head_dim、stride、连续性、device、dtype、尾块和空输入支持；目标不满足时仅在必要边界转换，保留合法回退。融合范围以所有消费者为界，不能丢弃 hook/graph/debug 仍需要的值。

量化和 KV 迁移分别追踪 K/V 的 scale、offset/zero-point 从 checkpoint 到 loader、cache 写入、Attention 消费、图捕获和 replay 的链路。混合 attention 模型只对适用层启用对应优化，不把 full-attention KV 优化扩展到线性/循环状态。

## 基线和测试

目标 baseline 固定目标权重、tokenizer/processor、采样设置、负载与参考算法；candidate 仅增加待验收迁移。涉及权重变换时核对 key 覆盖、形状、切分/合并和后处理，并验证变换前后等价性。

优先使用目标现有 goldens、算子参考和评测集。没有可运行参考时可以准备测试与补丁，但模型精度保持未验证；“输出可读”不是模型精度通过。

小规模先定位首次分歧：模块输出、Prefill logits、首个 Decode step、多个连续 Decode step、cache/状态更新，再按任务要求完成端到端评测。比较目标同模型 baseline/candidate 的数值误差和确定性输出；量化/非确定性路径使用项目已有标准，不临时放宽阈值。

单独验证变长 batch、长上下文、block 边界、滑窗截断、混合层、多模态 position 等受影响条件。测试数由契约与现有测试规范决定，不用无关随机 shape 数量替代关键边界覆盖。
