# 昇腾推理优化方案迁移 Skill

`ascend-infer-opt-migrate` 将已有模型推理优化迁移到另一昇腾服务器代际、另一模型或另一 vLLM-Ascend 版本，支持三种变化的组合。

## 覆盖场景

| 场景 | 模型 | 硬件 |
|---|---|---|
| H | 相同模型 | A2/A3/A5 跨代际 |
| M | 不同模型 | 同一服务器代际 |
| HM | 不同模型 | A2/A3/A5 跨代际 |

以上 H/M/HM 可保持框架版本不变，也可叠加 vLLM-Ascend 版本变化，形成 HV/MV/HMV；同模型、同硬件仅跨版本为 V。版本迁移覆盖升级、回迁、release/rc 和固定开发 commit；同时核对目标配套的 vLLM 与昇腾软件栈。

覆盖融合算子、推理框架、Prefill、Decode、内存、KV Cache、并行策略、调度策略和 PD 分离优化。

支持任意方向的迁移分析与适配流程；具体算子、API 和后端是否支持目标设备，以真实 SoC、目标版本文档与实测为准。

## 使用

在支持本地 Skill 的客户端中加载本目录后，提供源补丁/代码、目标工程与运行环境、模型配置、工作负载和验收要求。例如：

```text
使用 $ascend-infer-opt-migrate，将模型甲在 A3 上的融合算子和 Decode 优化迁移到 A5 上的模型乙。
源代码和目标工程路径如下……
保持目标模型的精度与接口，使用目标环境 baseline/candidate 对照验证性能。

使用 $ascend-infer-opt-migrate，将同模型、同 A3 上的优化从 vLLM-Ascend 源版本迁移到目标版本。
源/目标 vLLM-Ascend 与 vLLM 的 tag/commit 如下……先识别目标已内置的优化，再适配剩余接口差异。
```

Skill 会固定源优化契约和目标基线，分析硬件与模型差异，复用已有开发能力，完成最小适配，并按精度、性能和实际路径命中证据验收。缺少必要环境时交付可审阅改动及待验证项。

## 文件

- [SKILL.md](SKILL.md)：入口、场景判定、流程和完成条件。
- [hardware.md](references/hardware.md)：硬件能力、跨代方向和资源差异。
- [model.md](references/model.md)：模型结构、权重、量化、缓存和运行契约。
- [vllm-ascend-version.md](references/vllm-ascend-version.md)：版本配对、内置优化识别、接口映射、升级/回迁与目标版本验收。
- [optimizations.md](references/optimizations.md)：九类优化和已有 Skill 路由。
- [validation.md](references/validation.md)：目标端精度、性能、服务、PD 和回退验证。
- [pitfalls.md](references/pitfalls.md)：路径归属、算子契约探针、补丁落地、会话卫生、噪声底与日志限制。
- [openai.yaml](agents/openai.yaml)：客户端显示信息和默认调用提示。

可执行资产（模板，按目标工程改参数即可）：

- [operator-contract-probe.py](assets/operator-contract-probe.py)：退化输入探针（布局 / head 映射 / 是否内部归一化）。
- [bench-isolation.sh](assets/bench-isolation.sh)：隔离负载基准（prefill / decode / mixed），输出重复噪声。
- [noise-floor-compare.py](assets/noise-floor-compare.py)：噪声底与跨配置差异比较，给出“噪声内/超出噪声”判定。
- [migration-record.md](assets/migration-record.md)：最小迁移记录模板（含补丁自检与回退）。

## 依赖与验证边界

这是面向编码代理的指令 Skill，不包含自动执行的迁移引擎或硬件测试脚本。按实际任务使用目标工程、对应昇腾开发 Skill 和版本匹配的官方资料；可选 Skill 缺失时使用核实过的工程接口继续。

可执行资产是**模板**：算子签名、启动命令、指标名都需按目标工程改，不要直接当作通用驱动器。
已完成 Skill 格式和本地引用检查。`pitfalls.md` 及 `validation.md` 的归因/实验设计要求来自一次端到端实机迁移
（vllm-ascend PR #12647 的 GDN prefill 融合算子接入 A2 + Qwen3.8-Flash-Next w8a8，8×910B3）：
该案例中“改了文件不生效/日志不落盘/print 打断图编译/旧进程占显存/混合负载噪声大于效应”
都实际发生过，并已按本清单定位解决。版本轴（V/HV/MV/HMV）与其它硬件方向尚未实机验证。
编译通过、回退路径可运行和目标优化达到收益要求分别报告，不能互相替代。
源端历史收益不能替代目标端同环境对照数据。
