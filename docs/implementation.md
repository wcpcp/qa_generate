# ERP Data Generation Implementation Guide

## 当前实现主线

当前代码已经收敛成一条统一流程：

`metadata -> scene plan -> canonical QA -> task-aware postprocess -> final samples`

其中：

- `scene plan` 负责决定出哪些题
- `canonical QA` 负责规则真值
- `postprocess` 负责按任务策略决定是否调用模型做视觉增强、纠错或重包装

当前 object localization 主表达已经切到：

- `BFOV = [yaw, pitch, x_fov, y_fov]`

## 模块职责

### [schemas.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/schemas.py)

负责：

- `SceneMetadata / Entity / EntitySemantic` 的统一定义
- raw metadata 到运行时字段的归一化
- 提供 `resolved_xyz_camera`、`resolved_bfov`、`depth_bucket` 等便捷属性

### [pipeline.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/pipeline.py)

负责：

- metadata audit
- task feasibility
- anchor / partner 选择
- `scene_plan` 构建

当前 global task 也已经按新策略收紧：

- `existence` 默认只保留正例
- `counting` 只保留 `count>=2` 的正例

### [builders.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/builders.py)

负责：

- 把 `scene_plan` 转成 canonical QA
- 从问题模板和答案模板中稳定随机抽样
- 为不同任务生成结构化 `canonical_answer` 和可读的 `answer_text`

当前和最终版最相关的几条规则是：

- `grounding` / `direct_direction.precise`：输出 BFOV
- `distance_estimation.choice`：包含 object-reference 和 observer-reference 两类
- `relative_3d_position`：使用尺度感知的 `x/y` 阈值，以及 `|dz| >= 0.6m`
- `polar_distortion_awareness`：当前只保留 `shape` 恢复任务

### [postprocess.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/postprocess.py)

负责：

- 根据任务类型生成统一后处理 jobs
- 决定哪些样本进入模型、哪些直通
- 为视觉任务构建 ERP 图像消息
- 为文本任务构建规则事实重包装 prompt

当前默认策略：

- `counting`: 视觉纠错 + 重包装
- `caption`: 不进入后处理
- 其他规则任务：抽样文本重包装

当前统一默认抽样率是 `0.4`。

### [postprocess_execution.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/postprocess_execution.py)

负责：

- 执行后处理 jobs
- 合并模型输出
- 对 `counting` 接受纠错
- 对规则任务检查 `answer_core` 是否仍与 canonical 真值一致
- 对未抽样或失败任务按 fallback policy 回退到 canonical

### [providers.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/providers.py)

负责：

- 对接 OpenAI-compatible API
- 支持 `chat/completions` 和 `responses`
- 支持本地 cache
- 支持结构化 JSON 输出

### [orchestrator.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/orchestrator.py)

负责统一编排：

- `build_scene_bundle`
- `build_corpus_bundle`
- `execute_scene_bundle`
- `execute_corpus_bundle`

### [exporters.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/exporters.py)

负责导出：

- `scene_plan.json`
- `canonical_samples.jsonl`
- `postprocess_plan.json`
- `postprocess_jobs.jsonl`
- `postprocess_execution.json`
- `final_samples.jsonl`

## 推荐运行顺序

### 1. 检查 metadata

```bash
python3 data_generation/scripts/inspect_metadata.py \
  --input /path/to/scene_metadata.json
```

### 2. 看 scene plan

```bash
python3 data_generation/scripts/generate_scene_plan.py \
  --input /path/to/scene_metadata.json \
  --output /tmp/scene_plan.json
```

### 3. 看 canonical QA

```bash
python3 data_generation/scripts/generate_canonical_samples.py \
  --input /path/to/scene_metadata.json \
  --output /tmp/canonical_samples.jsonl
```

### 4. 跑统一主流程

```bash
python3 data_generation/scripts/build_training_data.py \
  --input /path/to/scene_or_directory \
  --output-dir /tmp/erp_training_bundle \
  --repackage-probability 0.4
```

### 5. 如需模型后处理，再打开 `--run-llm`

```bash
python3 data_generation/scripts/build_training_data.py \
  --input /path/to/scene_or_directory \
  --output-dir /tmp/erp_training_bundle_llm \
  --repackage-probability 0.4 \
  --run-llm \
  --base-url https://api.siliconflow.cn/v1 \
  --model Qwen/Qwen3.5-27B
```
