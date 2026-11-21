---
tags: [ASC, HPC, 注智体, WorldModel, UnifoLM-WMA]
type: 赛题整理
source: ASC26_Proposal_Ch5
---

# UnifoLM-WMA（Embodied World Model）赛题整理

> 题目：Embodied World Model — 具身智能世界模型推理加速
> 平台：8 × RTX 4090（24GB VRAM）+ 2 × AMD EPYC 7T83
> 环境：Miniconda3 + Python 3.10.18（未改动库版本，新增 Flash-Attention-2 需匹配版本）

---

## 一、赛题解读

### 1. 任务目标
UnifoLM-WMA（Unified Foundation of World Model and Action）是一个连接「高层环境感知」与「底层机器人控制」的具身智能框架：通过世界模型 + 动作条件控制，让智能体（如 Unitree G1 人形机器人、Z1 机械臂）在执行前做「mental rehearsal」——先在神经模拟器里推演未来状态，再付诸物理执行。

本赛题要求完成三件事：
1. **Baseline 部署**：在 5 个场景、20 个 case 上建立稳定推理环境。
2. **GPU 推理优化**：通过高阶注意力机制 + 编译技巧，最小化总执行时间。
3. **精度校验**：每个 case 保证 **PSNR ≥ 25**。

### 2. 推理流水线（三大功能模块）
- **Prompt Embedder**：对输入观测与动作 token 做特征提取，处理 `.csv` 元数据与环境 prompt，得到潜在表示。
- **Diffusion-Based World Model**：生成引擎。用扩散过程把高斯噪声迭代转化为连贯的 **16 帧视频序列**，模拟动作的物理后果。
- **Action-Conditioned Controller**：保证预测世界状态与物理控制向量的对齐，用 **PSNR（目标 ≥ 25）** 评估生成动作的物理精度。

### 3. 核心难点（瓶颈）
**① 迭代采样计算冗余高**
- 通过 `torch.profile()` 分析：**DDIM sampler 是最大延迟来源**。
- 50 步推断中，每步需对 U-Net 做 **两次前向**（条件 + 无条件引导）→ 共 **100 次完整 U-Net 前向**，占据绝大多数执行时间。
- 模型流水线强单向、依赖性强，**只能单 GPU 完成全流程**，无法多卡并行。

**②. 注意力机制受限于「存储墙」**
- 标准注意力随序列长度**二次方缩放**，HBM 与计算核之间频繁 I/O。
- 高分辨率、多头场景下 GPU 无法跑到理论峰值，baseline 平均实际执行时间 **> 11 分钟**。

---

## 二、优化思路

| 优化方向 | 核心思路 | 是否采用 |
|---|---|---|
| 注意力机制 | 用 **Flash Attention-2** 替换 `xformers` 的 memory-efficient attention | ✅（局部队放弃） |
| 预编译 | 用 **torch.compile** 预编译最耗时的 WMA 模型子模块 | ✅（局部队放弃） |
| 逻辑优化 | 修正 DDIM 采样两次调用都跑双 U-Net 的逻辑缺陷 | ✅ |
| 推理引擎 | 尝试 **TensorRT**（PyTorch→ONNX→engine）替代推理 | ❌ 放弃 |

**放弃 TensorRT 的原因：**
1. 主模型 WMA 并非纯 PyTorch 实现，转换耗时。
2. 测试 encoder/decoder 后，engine 占用过高内存，RTX 4090 的 24GB 无法同时载入原模型 + engine。

---

## 三、实现逻辑

### 1. 注意力机制优化（Flash Attention-2）
**原理**：Tri Dao 提出，两大核心技术绕过存储墙——
- **Tiling（分块）**：把整张注意力矩阵拆成小块，适配 SRAM 高速小缓存，逐块计算。
- **Operator Fusion（算子融合）**：把 matmul、Softmax、Mask 等融合为单算子，减少 HBM⇆缓存间中间结果往返。

效果：内存占用由二次增长降为**线性增长**，相比其他注意力机制快 2–4 倍。

**实现要点**：
- 重写 `flash_attn_forward`（由 `efficient_forward` 复制改造）。
- 核心难点在于数据类型与结构：需把 tensor 从 3 维 reshape 成 4 维，并从 **fp32 转成 bf16**；输出时再转回 **fp32**。
- 通过 online softmax + 分块计算，**避免显式构建 N×N 注意力矩阵**，峰值内存低于 xformers（尤其 320×512 高分辨率、64 多头场景）。
- ⚠️ 代价：频繁数据类型转换会引入精度损失（影响 PSNR，见下）。

### 2. 预编译优化（torch.compile）
**原理**：PyTorch 2.0 编译器，把 Python 解释执行的推理转换为直接调用**融合后的 CUDA kernel**，并将多次访存合并为一次，大幅减少执行时间。

**WMA 模型结构（条件时空 U-Net）**：
- 共享视频生成主干 + 动作预测头 + 状态预测头。
- 主干为带 skip connection 的 encoder-decoder；含 ResBlock（带 timestep embed）、`SpatialTransformer`（帧内自注意力 + 外部 context 交叉注意力）、`TemporalTransformer`（时间轴自注意力）。
- 动作/状态头接收主干中间特征图作为条件，进行 grounding。

**实现要点**：
- WMA 结构不适合整体 compile，**选择性编译不依赖 `isinstance()` 检查的子模块**：`action_unet`、`state_unet`、`time_embed`、`out_layer` 以及 transformer 内部组件。
- 配置：**增加缓存区大小**以减少重编译；**开启 TF32** 加速且几乎不损失精度。

### 3. 逻辑优化
**问题**：原代码 DDIM 采样被调用两次——
- 第① 次：预测动作序列；
- 第② 次：用动作序列生成状态与视频帧。

但**两次调用都把 `action_unet` 和 `state_unet` 都跑了一遍**，导致大量无效计算。

**实现**：条件判断当前目标，在运行中只计算对应的那个 U-Net（第① 次只算 action_unet，第② 次只算 state_unet），显著减少无效前向。

### 4. 精度（PSNR）权衡
- **Precompiling 与 FlashAttention-2 都会轻微拉低 PSNR**（重构了部分计算）；逻辑优化不影响 PSNR。
- 测试发现：第 4 数据集 `unitree_z1_dual_arm_stackbox_v2` 的 Case1 与 Case3 无法通过 PSNR≥25。
  - Case 3：**不用 prefuffle_precompiling**；
  - Case 1：**不用 Flash-Attention-2**。
- 通过按 case 选择性降级优化，保证全部 PSNR ≥ 25。

---

## 四、实验结果

### 时间优化（对 unitree_g1_pack_camera 的 case1 测试）
- 三种优化各自都有显著收益；
- 预编译还需从 cache 加载子模块（本身较耗时，故**真实推理加速比图上显示更高**）；
- 持续迭代后最终较 baseline 提速 **27%**（末次迭代）。

### 最终成绩（20 个 case）
| 项目 | 结果 |
|---|---|
| 提升幅度 | 4.4% ~ 15.0% 快于 baseline |
| 时序范围 | baseline 约 450~710s → 优化后约 423~627s |
| PSNR | 全部 ≥ 25（最低 25.17）|

（示例：Scene1 Case1：649.326s → 567.875s，提升 12.5%，PSNR 37.40；Scene5 Case1 提升最高 15.0%）

---

## 小结
- **提速借助三大招**：Flash Attention-2（内存高效注意力）、torch.compile 预编译子模块、DDIM 逻辑剪枝。
- 尝试过的 TensorRT 因内存不足与移植成本被迫放弃。
- 为保证精度阈，个别 case 主动放弃部分优化 → 取得 4.4%~15% 的一致提速，且 PSNR 全部达标。