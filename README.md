# pymss-ane

mel-band-roformer-deux (becruily) 的 ANE (Apple Neural Engine) 移植——M4 上以
单程序 MIL 全图编译运行，多档 T 全面快于 MLX 基线。

> 权重与配置不在仓库内（418MB）。需要：`weights/`（manifest.json 索引的 npz 权重，
> 由 `scripts/extract_weights.py` 从原始 checkpoint 生成）和模型配置 yaml
> （默认路径可用 `--config` 覆盖）。

## 快速开始

```bash
# 依赖：.venv（含 aneforge，uv sync 安装）、weights/、模型配置 yaml
uv sync

# 编译 + 推理单发（--config 指向你的 yaml）
.venv/bin/python scripts/build_graph.py --T 256 --config /path/to/config.yaml

# bench 矩阵（T=64/128/256/512 × p50/TFLOPS）
.venv/bin/python scripts/bench.py

# parity（numpy fp64 参考）
.venv/bin/python scripts/parity.py --T 512
```

## 性能（M4, N=30 p50, 2026-10-25）

| T | ANE | MLX (含STFT) | 加速 |
|---|---|---|---|
| 64 | 91.5ms / 2.49 TF | 262.8ms | 2.87x |
| 128 | 235.0ms / 1.99 TF | ~510ms | ~2.2x |
| 256 | 555.6ms / 1.77 TF | 960.7ms | 1.73x |
| 512 | 1648.6ms / 1.31 TF | 1906.1ms | 1.16x |

精度：T=64/256 d12 cosine ≥0.9994；T=512 d12 0.9883（fp16 十二层 softmax 累积界）。

## 关键实现点

- **T≥385 multi-input bandsplit**：单 input 被 ≥13 组 slice 消费 + T≥385 触发
  Espresso 编译墙（384=3×128 tile 对齐豁免）。修复 = 39 个独立 per-group input，
  host 按组边界预切 L 维。详见 scripts/bs_bisect.py 的二分定位。
- **softmax 精度**：row-sum 用 @ones 宽累加器（af.reduce_sum_to_matmul），代价
  23-39% 延迟；可按需回退 narrow reduce_sum（精度 T=256 0.9965）。
- exp2 softmax（原生 exp 不能喂 bmm）、rope 4D x@P（65536 轴上限）、
  big_linear flatten、conv1x1 评估结论（transpose 成本>收益，未采用）。

## 布局契约（host ↔ ANE）

- 输入 x: [1, T, 7916]，group-major band 布局（T≥385 时为 39 个 per-group 切片）
- 输出: [T, 15832] = [S=2 × L=7916] masks，stem-major 行序（host reshape 回 [S,T,L]）

## 报告

性能/根因/瓶颈全景：`../.pi/herdr-team/6kq413/reports/task19-perf-final-delivery.md`
