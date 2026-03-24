# ERP Data Generation

当前版本已经收敛成一条更简单的主流程：

`metadata -> scene plan -> canonical QA -> task-aware postprocess -> final samples`

这里不再保留旧的 `QC -> verification -> realization` 三段式后处理。
原因很直接：

- 纯启发式 QC 不稳，价值不高
- 不是所有任务都需要验证
- 百万级数据不可能全部逐条交给大模型重包装

所以现在采用的是：

- `counting`：必须走视觉验证、纠错和问答重包装
- `caption`：保持 canonical，不进入统一后处理
- 其他规则真值任务：默认只抽样做重包装，不做验证
- 非抽中的样本直接保留 canonical 版本

同时，当前物体在 ERP 球面空间中的主定位表达已经切到：

- `BFOV = [yaw, pitch, x_fov, y_fov]`

主流程不再把单个中心点角度当成完整物体定位表达。

## 当前原则

### 1. canonical 层负责真值

[generate_canonical_samples.py](/Users/wcp/code/erp_data_pipeline/data_generation/scripts/generate_canonical_samples.py)
和 [builders.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/builders.py)
负责规则化生成 canonical QA。

这一层：

- 不依赖外部模型
- 尽量使用多套 question / answer templates
- 保证答案来自 metadata 或规则计算

### 2. postprocess 层负责任务感知的后处理

[postprocess.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/postprocess.py)
和 [postprocess_execution.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/postprocess_execution.py)
负责统一后的模型后处理。

这一层：

- 不再把验证和改写拆成两套体系
- 每个任务 family 自己决定要不要用模型
- `counting` 可以纠错
- 其他任务只允许改写，不允许改真值

### 3. 视觉输入当前默认用 ERP 图

当前 metadata 里有：

- `image_path`
- `representative_view_id`
- `source_views`
- `local_reground.view_id`

但没有稳定可直接读取的透视图文件路径。

所以当前实现默认是：

- `counting` 直接使用整张 ERP 图
- 其他进入后处理的规则任务主要依赖结构化事实
- object localization 主表达优先使用 `BFOV`

后面如果你们补充了透视图路径映射，可以继续把 representative view crop 接进来。

## 当前任务策略

### 需要视觉后处理

- `counting`
  - 模型看 ERP 全图
  - 必须重新核查计数
  - 如果原答案错误，允许纠正
  - 如果无法确认，直接过滤

### 不做验证，只做抽样重包装

- `existence`
- `grounding`
- `direct_direction`
- `relative_direction`
- `view_transform`
- `distance_estimation`
- `relative_3d_position`
- `seam_continuity`
- `polar_distortion_awareness`

这些任务的真值主要来自规则和几何计算，因此模型只负责：

- 换一种问法
- 适度增加答案表达多样性
- 对部分关系类问题增加简短解释

其中：

- `caption` 当前直接保留 canonical，不进入统一后处理
- 其他列出的规则任务默认只抽 `40%` 做这一步，可通过参数改

其中 `existence` 现在重新保留了 metadata-level 负例，但负例会在后处理 prompt 里基于更大的缺失类别池重新包装，而不是只死用初始候选。

## 统一入口

现在推荐只用一个主脚本：

- [build_training_data.py](/Users/wcp/code/erp_data_pipeline/data_generation/scripts/build_training_data.py)

它统一负责：

1. 读取 metadata
2. 生成 `scene_plan`
3. 生成 `canonical_samples`
4. 构建 `postprocess_jobs`
5. 可选执行 LLM 后处理
6. 导出最终结果

## 快速开始

### 1. 检查 metadata

```bash
python3 data_generation/scripts/inspect_metadata.py \
  --input data_generation/dataset/metadata.json
```

### 2. 单独查看 scene plan

```bash
python3 data_generation/scripts/generate_scene_plan.py \
  --input data_generation/dataset/metadata.json \
  --output /Users/wcp/code/erp_data_pipeline/data_generation/results/scene_plan.json
```

### 3. 单独查看 canonical QA

```bash
python3 data_generation/scripts/generate_canonical_samples.py \
  --input data_generation/dataset/metadata.json \
  --output /Users/wcp/code/erp_data_pipeline/data_generation/results/canonical_samples.jsonl
```

### 4. 统一生成训练数据包，不执行 LLM

```bash
python3 data_generation/scripts/build_training_data.py \
  --input data_generation/dataset/metadata.json \
  --output-dir /Users/wcp/code/erp_data_pipeline/data_generation/results/training_bundle \
  --repackage-probability 0.4
```

这一步会导出：

- `postprocess_plan.json`   这个是非常全的，包括jobs，summary，和一些其他统计
- `postprocess_jobs.jsonl`
- `summary.json`
‘’‘
canonical_sample_count = 17    这条 scene 最终生成了 17 条 canonical QA。

postprocess_job_count = 8    这 17 条里，有 8 条被选中进入后处理。
后处理的含义是：
counting：视觉验证 + 纠错 + 重包装
其他被抽中的规则任务：只做重包装

passthrough_count = 9   有 9 条没有进入后处理，直接保留 canonical 版本。
这通常包括：
没被抽中做重包装的规则题
当前策略明确不进后处理的题，比如 caption

filtered_postprocess_count = 0     本来打算进入后处理的样本里，没有任何一条被直接判定为“必须过滤掉”。

skipped_postprocess_count = 7     有 7 条是“本来属于可后处理任务”，但这次没有真正进入 job 列表。
在你当前这版里，这基本等价于：
没被 0.4 概率抽中
所以被记成 skipped
更直白一点，你这次这条 scene 的流转是：

先生成 17 条 canonical 样本
其中 8 条进入 postprocess jobs
另外 9 条直接 passthrough
没有样本被强制过滤
那 7 条 skipped，其实是 9 条 passthrough 里的一个子集来源：
它们本来是“可重包装任务”
但因为抽样没抽中，所以跳过了 postprocess
剩下的 passthrough 则更多是像 caption 这种策略上就不进 postprocess 的题
’‘’

默认不再重复保存 `scene_plan.json` 和 `canonical_samples.jsonl`，
因为它们已经可以分别通过：

- [generate_scene_plan.py](/Users/wcp/code/erp_data_pipeline/data_generation/scripts/generate_scene_plan.py)
- [generate_canonical_samples.py](/Users/wcp/code/erp_data_pipeline/data_generation/scripts/generate_canonical_samples.py)

单独得到。

### 5. 统一生成并执行 LLM 后处理

```bash
export SL_KEY=...

python3 data_generation/scripts/build_training_data.py \
  --input data_generation/dataset/metadata.json \
  --output-dir /Users/wcp/code/erp_data_pipeline/data_generation/results/training_bundle_llm \
  --repackage-probability 0.4 \
  --run-llm \
  --base-url https://api.siliconflow.cn/v1 \
  --model Qwen/Qwen3.5-27B
```

如果模型调用成功，还会额外导出：

- `postprocess_execution.json`
- `final_samples.jsonl`

## 当前输出文件含义

### `postprocess_plan.json`

记录本次后处理策略、抽样结果、哪些样本直通、哪些样本进入模型。

### `postprocess_jobs.jsonl`

真正准备发给模型的任务列表。

### `final_samples.jsonl`

最终训练样本。

其中既包括：

- 直通的 canonical 样本
- 经过 LLM 重包装的样本
- `counting` 的纠错结果

## 代码入口

- [schemas.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/schemas.py)
  负责 raw metadata 到运行时结构的归一化

- [pipeline.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/pipeline.py)
  负责 scene plan

- [builders.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/builders.py)
  负责 canonical QA

- [postprocess.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/postprocess.py)
  负责任务感知的后处理计划与 prompt 构造

- [postprocess_execution.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/postprocess_execution.py)
  负责执行模型输出并合并为最终样本

- [providers.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/providers.py)
  负责对接 OpenAI-compatible API

- [orchestrator.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/orchestrator.py)
  负责统一编排

- [exporters.py](/Users/wcp/code/erp_data_pipeline/data_generation/src/erp_data_generation/exporters.py)
  负责导出 scene / corpus 结果

## 目前最重要的配置

- [question_templates.json](/Users/wcp/code/erp_data_pipeline/data_generation/templates/question_templates.json)
- [answer_templates.json](/Users/wcp/code/erp_data_pipeline/data_generation/templates/answer_templates.json)
- [postprocess_policy.json](/Users/wcp/code/erp_data_pipeline/data_generation/config/postprocess_policy.json)

## 当前已知边界

- 当前视觉后处理默认依赖 ERP 原图路径可访问
- 当前只有 `counting` 强制依赖视觉输入
- 如果图像路径不存在，`counting` 会被过滤，不会偷偷回退成 canonical
- 当前不再生成 `zero-count counting`
- `relative_3d_position` 当前采用“尺度感知的 x/y 阈值 + 绝对 z 阈值 0.6m”
- `distance_estimation.choice` 当前包含两类 reference：
  - 物体 reference：基于 center-to-center 3D distance
  - observer reference：基于深度比较，并且默认采样概率更高
