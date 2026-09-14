# 让 GPU Benchmark 学会说“不”：Kairo 的可复现实验门槛

> 草稿状态：框架。面向推理工程与性能工程读者。

## 开场：快一点不等于赢了

GPU 性能结果最常见的问题不是数字算错，而是比较对象变了：不同模型 revision、
不同实际 token 数、不同 warm-up、不同 CUDA Graph 设置，或者服务根本没有正确回答。

Kairo 把“能否发布一个性能结论”当成工程接口，而不是写作时的自律。

## 一条结果必须回答的六件事

1. 哪个模型与权重 revision？
2. 哪个硬件、驱动、CUDA、框架和源码 revision？
3. 相同请求实际处理了多少输入、输出 token？
4. 唯一改变的变量是什么？
5. 结果是否通过数值或语义正确性门？
6. 原始结果、命令和汇总表能否对应？

## Kairo 的实验协议

用一段精简 YAML 解释：workload、runtime、correctness、repeat、raw artifact hash 与
promotion gate。避免贴大段配置；读者需要理解约束，而不是背字段名。

## 三个“没有被提升”的结果

### 1. Graph 并非总是大幅领先

27B、c16、2K prompt/4K context 只有 1.30×。它仍是正收益，却不足以支持“到处开
Graph”的结论。

### 2. SGLang 的 Mamba 内存预算不是全局旋钮

短 prompt c16 的 ratio 8 有约 +71% 吞吐；2K prompt 时收益消失。把有限显存从 KV
cache 挪给状态缓存，本质是 workload trade-off。

### 3. TMA swizzle 通过编译也不代表正确

`m256_s32` 在与 cuBLAS 的比较中误差达到 18.52，因此被明确拒绝。这不是“调参未完”，
而是 shared-memory layout contract 不匹配。

## 把负结果保存下来有什么用

- 防止同一条死路被反复尝试。
- 让策略边界能够被解释。
- 让下一位贡献者知道哪些结论已经被证伪、在哪些条件下成立。

## 可复现不等于绝对客观

说明仍存在的边界：单机、WSL、夜ly runtime、少量 workload buckets、语义 smoke
不是模型质量评估。透明地限定结论，比扩大标题更能获得技术信任。

## 读者可以复跑什么

给出最短路径：验证 YAML、运行 portable test、查看公开结果、再在同类 GPU 重跑服务协议。

<!-- 发布前：提供 public-results/，放入 JSONL、环境锁定文件与生成表格脚本。 -->
