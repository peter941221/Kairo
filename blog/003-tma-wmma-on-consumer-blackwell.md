# 在消费级 Blackwell 上追一条 TMA + WMMA 内核：进展、边界与一次失败

> 草稿状态：框架。面向 CUDA 开发者；尽量少用未解释的术语。

## 开场：为什么还要写自研 GEMM

vLLM 和 CUTLASS 已经很快。自研内核的目的不是立刻取代 cuBLAS，而是建立一条可验证的
Blackwell 数据搬运与矩阵计算路径，找出真正值得投入的缺口。

## 先确认硬件实际支持什么

- RTX 5090 是 SM120；本地 CUDA 13.0 能运行异步 shared-memory copy。
- 真实 2D TMA copy 已通过，最大绝对误差为 0。
- 当前工具链的 `tcgen05` 探针不可用；因此不把它写成“已支持”的前提。

一句解释：TMA 用于把大块数据搬到 shared memory；WMMA 用 Tensor Core 做小矩阵乘法。

## 从标量 tile 到 TMA + WMMA

| 里程碑 | 结果 | 含义 |
|---|---:|---|
| 标量 tiled FP16 | 22.5 TFLOP/s @ 1K | 正确的基础实现 |
| 直接 WMMA | 约 33.3 TFLOP/s @ 1K | Tensor Core 带来跃升 |
| TMA + WMMA | 约 35.2 TFLOP/s @ 1K；41.6 TFLOP/s @ 4K | 数据搬运路径闭环 |
| m128/m256 shape policy | 4K 最多约 49.9 TFLOP/s | tile 选择依赖 shape |

<!-- 发布前：统一使用稳定重复结果，补 cuBLAS 对照、误差、迭代数与完整 shape 表。 -->

## 最重要的结果：没有“万能 tile”

更大的 block 在大矩阵上可能获益，小 shape 却会输给 64-row control。因此 Kairo 不把
一个 benchmark 冠军设为全局默认，而是只为实测 cell 提升候选变体。

## 一次有价值的失败：swizzle

`m256_s32` 使用 TMA swizzle 后的结果对 cuBLAS 出现 18.52 的最大绝对误差。原因不是
“GPU 不支持 swizzle”，而是生产端写入的布局与消费者 WMMA 的 row-major 读取假设不一致。

这个失败说明：异步搬运、shared-memory layout 和矩阵指令必须作为一个整体设计。

## 为什么这还不是性能宣言

即使正确，当前实现仍明显慢于 cuBLAS。文章应明确把它称为“实验内核”，并将下一个问题
限定为 occupancy、layout、staging 与可用 Blackwell 指令路径，而非模糊地说“继续优化”。

## 可复现与下一步

链接到 capability probe、TMA copy、GEMM probe、协议和测试；列出下一步：补 boundary
coverage、配套 swizzled 消费布局、以及在有可用工具链时重新评估 native 指令路径。
