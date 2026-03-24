# ERP Data Construction Framework

## 目标

当前框架的目标很明确：

从你们已经处理好的 ERP scene metadata 出发，构建适合训练 ERP 基础模型的高质量 QA 数据。

核心原则有三条：

1. 真值尽量来自 metadata 和规则，不让 LLM 决定大多数任务的 GT
2. 不把所有样本都送进 LLM，而是按任务类型决定是否需要模型参与
3. 让模板多样化成为规模化主力，让 LLM 只承担少量高价值后处理

## 当前数据现实

你们当前的 processed metadata 已经很丰富，至少包含：

- ERP 图像路径
- ERP 中的实体区域
- `bfov = [yaw, pitch, x_fov, y_fov]`
- `lon_lat`
- 语义标签、属性、caption
- local reground 结果
- 深度和 `xyz_camera`
- representative view 标识

所以这套框架不再把自己建立在“极简 metadata”上，而是承认：

- 你们已经有较强的基础元数据
- 后面真正该优化的是 QA 构造和后处理

## 任务层设计

### 1. 基础理解

- `caption`
- `existence`
- `counting`
- `grounding`
- `scene_understanding`

其中：

- `caption` 统一承接 identify / attribute / description
- `scene_understanding` 目前只保留名字，不作为主流程任务

### 2. 全向理解

- `direct_direction`
- `relative_direction`
- `view_transform`

### 3. 3D 空间理解

- `distance_estimation`
- `relative_3d_position`

### 4. ERP-specific

- `seam_continuity`
- `polar_distortion_awareness`
- `rotation_consistency`

其中：

- `rotation_consistency` 目前只预留，不进入主流程

## 当前后处理策略

### `caption`

当前直接保留 canonical，不进入统一后处理。

原因是：

- `caption` 的答案本身来自 `caption_dense`
- 这类题面后面更适合单独做高质量 LLM authoring，而不是混在统一重包装主流程里
- 当前主流程里优先把模型预算留给 `counting`

### `counting`

必须走模型，因为计数最容易受到漏检影响。

这里模型负责：

1. 重新观察 ERP 图像
2. 判断原 count 是否正确
3. 如果错误则纠正
4. 再输出包装后的 QA

### 其他规则任务

默认不做验证，只做抽样重包装。

原因是这些任务的真值本身来自：

- bbox
- yaw / pitch
- depth
- xyz
- 几何规则

所以没必要把所有样本都再交给模型重新判断。

当前统一默认抽样率是 `0.4`。

同时 object localization 主表达已经切到：

- `BFOV = [yaw, pitch, x_fov, y_fov]`

而不是只用中心点角度。

## 当前视觉上下文策略

当前 metadata 里虽然有：

- `representative_view_id`
- `source_views`
- `local_reground.view_id`

但没有稳定直接可读取的透视图文件路径。

因此当前实现现在采用：

- `counting`：直接用 ERP 全图
- 其他任务：默认不依赖视觉输入

后面如果补充了透视图资产路径映射，还可以进一步把 representative view 原图直接接入。

## 为什么不再保留旧的 QC

因为纯启发式 QC 在这里不是最关键的瓶颈。

当前更有价值的是：

- 问题模板足够多样
- 答案模板足够多样
- `counting` 做纠错
- 其他任务只在必要时做重包装

所以旧的 heuristic QC 已经从主流程移除。

## 现在最重要的扩展方向

1. 持续扩充 question / answer templates
2. 让 `counting` 的视觉 prompt 更稳
3. 持续优化按任务拆分的后处理 prompt
4. 把少量高质量 LLM 输出持续回流成模板
