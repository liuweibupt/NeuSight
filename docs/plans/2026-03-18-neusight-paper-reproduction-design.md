# NeuSight 论文复现与 A100/GPT-3 Prefill/Decode 扩展设计

**日期：** 2026-03-18  
**仓库：** `NeuSight`  
**工作分支：** `paper-repro-a100`

## 1. 目标

本次工作分两阶段进行：

1. **先复现论文中仓库已经明确支持的结果**，并尽量最大程度依赖本地已有内容。
2. **在论文复现完成后做最小改动**，补出用户需要的 A100 上 GPT-3 Prefill/Decode 数据路径。

这里的“论文复现”优先定义为：
- 使用仓库现有的 predictor、label、脚本与汇总脚本；
- 重新生成 ASPLOS 目录下的 prediction / summary 结果；
- 重点核对 A100 相关结果；
- 允许与仓库已有 summary 存在小幅波动，仓库 README 已说明预测结果可能存在约 10% 的非确定性差异。

## 2. 范围

### 2.1 第一阶段：论文复现

第一阶段只覆盖仓库和 README 明确支持的流程：
- `scripts/asplos/run_pred_nvidia_neusight.py`
- 必要时配套的 NVIDIA baseline prediction 脚本
- `scripts/asplos/summarize.py`
- `scripts/asplos/table.py`

重点关注：
- NVIDIA 平台结果是否可以重新生成；
- A100 相关模型结果是否与仓库内 summary 接近；
- 输出文件结构是否与仓库现有目录约定一致。

### 2.2 第二阶段：最小改动扩展

第二阶段在不大改 NeuSight 主体结构的前提下，补出用户需要的数据：
- **Prefill**：`batch_size=1, seq_len=128, d_model=12288, n_heads=96, data_type=fp16, device_count=1`
- **Decode**：`batch_size=1, seq_len=2048, d_model=12288, n_heads=96, data_type=fp16, device_count=1`

第二阶段优先复用：
- 现有 `scripts/pred.py`
- 现有 tracing / parsing / aggregation 流程
- 现有 A100 device config
- 新增尽可能少的 model config、执行模式和辅助脚本

## 3. 现状判断

从仓库代码可见：
- 当前公开入口支持的 execution type 是 `inf` 与 `train`；
- tracing 中将 Hugging Face config 的 `use_cache` 关闭，现有推理路径更接近“整段 forward inference”；
- 仓库并没有显式的 `prefill` / `decode` 模式，也没有现成 KV-cache decode tracing 入口；
- 仓库内已经包含论文使用的 predictor、label、results、summary，因此最合理的复现起点是“结果级复现”，而不是强行从采集和训练全链路重做。

因此本次设计采用：
- **先做结果级论文复现**；
- **再做最小改动扩展到 prefill/decode**。

## 4. 方案选择

### 方案 A：结果级复现（采用）
直接依赖本地已有 predictor、label、脚本和 summary 入口，重跑 prediction / summarize / table。

**优点：**
- 最符合“最大程度依赖本地内容”；
- 最接近仓库作者提供的官方复现路径；
- 成本最低、结论最稳。

**缺点：**
- 不等于从零训练或从零采集数据的完整复现。

### 方案 B：训练级复现（暂不作为第一阶段目标）
重训 predictor 后再做 prediction 和 summary。

**优点：** 更接近完整论文流程。  
**缺点：** 对环境、时间和依赖要求更高。

### 方案 C：全链路复现（不采用）
从采集、处理、训练到预测全部重做。

**优点：** 最完整。  
**缺点：** 当前单机环境很难覆盖论文涉及的多种 GPU 平台。

## 5. 架构与执行路径

### 5.1 论文复现路径

1. 检查现有 ASPLOS 数据目录、预测器目录、标签目录和结果目录。
2. 在隔离 worktree 中运行论文脚本，优先重跑 NVIDIA NeuSight prediction。
3. 运行 `summarize.py` 与 `table.py` 重新生成汇总。
4. 将新生成结果与仓库已有 `summary/*.csv` 做对比。
5. 输出复现结论，尤其是 A100 结果。

### 5.2 Prefill/Decode 扩展路径

1. 先确认现有 `inf` tracing 如何构造 GPT 模型与 sequence length。
2. 新增一个与 GPT-3 目标参数匹配的 model config。
3. 在 predictor 入口增加最小扩展，使其能区分：
   - prefill：整段 prompt 的一次性 forward；
   - decode：带 cache 的单 token step，或在代码无法完整表达时给出与论文/实现一致的近似建模路径。
4. 为 A100 设备和用户参数生成对应输出文件。
5. 单独报告 prefill 和 decode 结果，不混入第一阶段论文复现结论。

## 6. 数据流

### 第一阶段
`device_config + model_config + predictor weights + tile dataset`
→ `pred.py / run_pred_*`
→ `results/prediction/*`
→ `summarize.py`
→ `summary/*.csv`
→ 与仓库已有结果比对

### 第二阶段
`A100 device config + new GPT-3 config + new execution mode/minimal extension`
→ tracing/parsing/prediction
→ 生成 prefill / decode 输出
→ 提取用户要求参数的结果

## 7. 错误处理与回退策略

- **依赖缺失**：先记录环境问题，再尽量利用仓库已有结果与脚本推进文档化和可重跑流程。
- **脚本运行失败**：优先局部修复脚本兼容性，不重写整套流程。
- **数值与仓库 summary 存在偏差**：以 README 中“约 10% 波动”作为容忍参考，给出具体偏差说明。
- **prefill/decode 无法直接通过现有结构表达**：允许做最小实现补充，但避免破坏现有 `inf/train` 逻辑。

## 8. 测试与验证策略

### 第一阶段验证
- 相关 prediction 脚本可运行；
- 生成的 prediction/summary 文件路径正确；
- A100 条目可从 summary 中提取；
- 与仓库已有 summary 对比后给出一致/接近/偏离结论。

### 第二阶段验证
- 新模式不会破坏已有 `inf/train`；
- 能对用户指定参数成功生成结果；
- 输出中明确区分 prefill 与 decode；
- 给出最终 A100 + GPT-3 参数结果。

## 9. 非目标

本次工作暂不追求：
- 多 GPU 平台的从零数据采集；
- 全 baseline 从零训练；
- 对论文全部图表做全量重做；
- 对 NeuSight 模型架构进行系统性重构。

## 10. 交付物

本次工作最终应产出：
- 一份论文复现结论；
- 必要的复现脚本或兼容性修改；
- 重新生成的 summary / comparison 输出；
- 一条最小改动路径，用于得到用户指定的 A100 + GPT-3 prefill/decode 数据；
- 与之对应的 Git 提交记录。
