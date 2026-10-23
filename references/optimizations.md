# 九类优化：迁移范围、路由与证据

先选择用户涉及的行，再读相关 skill。以运行时可见的已安装 skill 清单定位，读取对应 `SKILL.md`；不将本机绝对路径写入待迁移项目，不假设可调用其中提到的所有工具。

九类优化均可跨 vLLM-Ascend 版本迁移。含版本变化时先读 [版本迁移指南](vllm-ascend-version.md)，对照目标已内置能力、vLLM 配对与接口差异，再执行下表的类型专项适配；版本迁移不是新增一种优化类型。

| 优化类型 | 迁移时核对与改动 | 可复用的已有 skill | 验收重点 |
|---|---|---|---|
| 融合算子开发 | 子图语义、所有消费者、shape/dtype/layout、注册/绑定、tiling、数值和同步 | `model-infer-fusion`；符合方向时 `ascendc-cross-gen-port` / `ascendc-cross-gen-port-light`；按实现语言用 AscendC/Triton/TileLang 开发与验证 skill | 实际命中、算子精度、完整模型、目标关键路径与 E2E |
| 推理框架优化 | runner/model 接口、后端注册、权重/metadata、图捕获/replay、编译缓存、服务入口 | `model-infer-migrator`（仅补目标部署基线）；`model-infer-graph-mode`；`torch-custom-ops-guide` | 同框架版本的 eager/图路径、接口兼容、稳态复用与服务负载 |
| Prefill 阶段优化 | packed/变长输入、mask、position、chunk 边界、KV 增量写入、融合/通信策略 | `model-infer-fusion`、`model-infer-kvcache`、`model-infer-profiling` | Prefill 精度、TTFT/阶段耗时、输入吞吐、长短上下文与 Decode 接续 |
| Decode 阶段优化 | decode batch、KV 长度/地址变化、图 bucket、循环状态、权重预取、通信依赖 | `model-infer-graph-mode`、`model-infer-prefetch`、`model-infer-multi-stream` | 连续多步精度、TPOT/ITL、p95、输出吞吐、capture/replay |
| 内存优化 | weight/KV/workspace/graph pool 峰值、分配与释放、alias、复用/预取/offload 生命周期 | `model-infer-kvcache`、`model-infer-prefetch`；GE 场景用 `ge-memory-analysis` | 实际峰值与稳态内存、请求退出后的回收、OOM 边界及搬运开销 |
| KV Cache 优化 | 连续/分页/MLA/混合层/滑窗、block/slot、量化 scale/offset、prefix 复用、图地址 | `model-infer-kvcache`、`model-infer-quantization`、`model-infer-precision-debug` | 写读等价、长序列和 block 边界、页复用/回收、容量、精度和 TPOT |
| 并行策略优化 | TP/EP/DP/PP/CP 与实际卡数/拓扑、权重 shard、heads/expert 整除、collective 顺序 | `model-infer-parallel-analysis`、`model-infer-parallel-impl`、`ascendc-mc2-best-practice` | 全 ranks 输出/退出、通信组与重排、模型精度、吞吐/延迟/内存 |
| 调度策略优化 | continuous batching、chunk、准入/优先级、token budget、preemption、请求状态、KV 配额 | 目标框架现有 scheduler、接口文档和测试；用 `model-infer-profiling` 辅助归因 | 混合长短请求、排队/p95、SLO 内吞吐、公平性、取消/超时/抢占后的资源回收 |
| PD 分离场景优化 | P/D 角色、KV 传输协议与布局、层/shard 映射、元数据、connector、网络与路由 | 目标框架现有 connector/transfer 实现和集成测试；KV/并行 skills 仅用于其实际覆盖子问题 | 本地基线到分离链路的精度、交接时延、TTFT/TPOT、SLO 吞吐、失败后回收 |

## 路由边界

- 环境核对可使用 `ascendc-env-check`；已有远程执行工具链可用 `remote-cann-development`，不能为了迁移强制新增一套远程后端。
- 性能采集复用现有框架 profiler；必要时用 `model-infer-profiling`、`ops-profiling`。拆解可用 `model-infer-perf-breakdown`，原始 trace 保留阶段、rank/device、shape 与采样边界。
- 已安装的跨代算子 skill 有方向和依赖限制，见硬件资料。任意方向均可分析，不代表专用引擎支持任意方向。
- `model-infer-*` 中部分约定针对 cann-recipes-infer 单机/BF16 或特定部署模式；用于 vLLM-Ascend 或其他框架时，只复用适用技术指导，核对真实接口，不照搬目录、默认缓存模式或配置字段。
- `model-infer-graph-mode` 的阶段约定不能代替目标框架支持矩阵；Prefill/Decode 能否用图取决于目标真实路径及验证。
- 未提供独立调度迁移或 PD 全流程 skill 时，直接基于目标实现完成适配；不得虚构 skill 名或把单机并行分析当作多机 PD 的实现依据。
- 缺少可选 skill 时使用已核实的目标工程/官方文档继续；只有必要资源缺失且无可行替代时，将该项标记阻塞。

## 多项优化的组合

按实际依赖实施：权重/并行布局变化可能影响融合输入、KV 结构和 PD 传输；缓存与图地址约束可能影响调度和抢占；调度的 batch/chunk 变化可能改变 Prefill/Decode 路径。

优先逐项对照与最终组合验证，记录各次启用项。组合比单项退化时检查带宽竞争、通信重叠、capture bucket、workspace 峰值和调度等待。用户仅要求迁移指定优化时，不自行增加新的量化、并行或调度策略。

PD 的 P/D 模型权重、tokenizer 和请求语义必须一致；跨模型迁移是把方案迁入新的模型服务，不是把模型甲产生的 KV 直接供模型乙解码。目标端 P/D 硬件是否同代、传输是否可跨硬件，以用户部署及后端证据为准。
