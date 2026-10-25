# ANE 性能复核：已测现象、未证机制与候选优化

本轮起点：复核先前所谓“ANE↔GPU 结构性互扰税”，并直接测量实际模型矩阵形状与密集计算峰值的差距。以下以落盘 JSON 为证据，不沿用旧报告中的硬件因果推测。

## 结论摘要

1. **没有在受控测试中复现固定的 ANE→GPU 惩罚。** 8 轮随机交错测试中，GPU-after-GPU 中位数 16.500ms，GPU-after-ANE 16.507ms，比值 1.000。该结果不证明所有配置都没有干扰，但不支持先前“固定 +15–21ms/对”的普遍结论。
2. **调用路径存在已确认的转换/拷贝成本。** `Model.__call__` 把输入复制为 fp16，输出读回并转换为 fp32；原混合代码随后又转回 fp16。使用 ANE 缓冲区视图后，T512 单层链路从 84.457ms 降到 71.891ms，输出逐位一致。MLX `mx.array(numpy)` 本次存储共享检查为 False，不能称为零拷贝。
3. **当前 ANE 跑道可接近密集计算峰值。** conv1×1 链 16.34 TFLOPS，linear 链 14.36 TFLOPS；实际 attention 的批量小矩阵远低于该吞吐。不能把整图 1–3T 解释为硬件所有算子都只有该速度。
4. **已测局部优化没有形成可靠整图收益。** batch/head 打包独立 QK/AV 较快，但完整 attention 和小档整图更慢；RoPE 连续半块方案在小档整图只有约 0.44% 的中位差，尚不足以证明稳定收益。生产 `build_graph.py` 保持原样。

## 1. 受控交替测试

脚本：`scripts/diagnose_ane_gpu.py`
证据：`.tmp/ane_gpu_repro.json`

真实深度0 PM2 图，T512、实际模型 heads=8、dim_head=64；GPU 随机输入提前求值。直接记录每个边界耗时，随机交错顺序，不用“总时间减独立基线”归因。

| 被计时部分 | 前置操作 | median ms |
|---|---|---:|
| GPU SDPA | GPU SDPA | 16.500 |
| GPU SDPA | ANE PM2 | 16.507 |
| ANE PM2 | ANE PM2 | 38.824 |
| ANE PM2 | GPU SDPA | 40.574 |

没有 GPU 频率、功耗、Metal 内核时间或带宽计数器证据支持“电源域恢复”“rush-current”“硅片物理定律”的解释。旧报告同时存在 19.8→69.5ms 与 +15–21ms 的口径不闭合，以及跨越次数/交替对的混用，均不宜继续作为定量依据。

## 2. 真实数据交接审计

脚本：`scripts/diagnose_ane_gpu_bridge.py`
证据：`.tmp/ane_gpu_bridge_audit.json`

| T512 单层 PA→GPU→PM2 | median ms | min ms |
|---|---:|---:|
| 原公共调用路径 | 84.457 | 78.509 |
| fp16 视图路径 | 71.891 | 66.488 |

8 组配对均为视图路径较快。公共 API / 手工逐段封装 / 视图路径输出逐位一致。所有计时段之和与总账差小于 0.1ms。

原路径的 QKV 输出读取、fp32 转换、再转回 fp16，分段中位数分别约 2.45、3.47、4.07ms。MLX 导入仍约 2–3ms。NumPy 导出与 contiguous 本轮几乎无额外耗时，不能沿用先前对这两处的拷贝成本猜测。

### 完整模型对照

脚本：`scripts/diagnose_hybrid_e2e.py`

T64/d2：原混合 38.716ms，视图混合 31.547ms，全 ANE 21.097ms；混合相对 NumPy fp64 的 cosine >0.999，原/视图混合逐位一致。

T512/d12（8 轮同进程随机交错、每深度真实独立权重、相同 reduce-fix 配置）：

| 路径 | min ms | median ms | max ms |
|---|---:|---:|---:|
| 原混合 | 1882.388 | 3794.348 | 11553.044 |
| 视图混合 | 1179.447 | 1738.565 | 2846.477 |
| 全 ANE | 1078.856 | 1402.755 | 2678.316 |

当前视图实现仍未赢过全 ANE。这轮长测有明显波动，不能据此推断固定硬件税或排除未来其他混合实现。测试后系统快照存在约 2.37GB swap 占用；没有此长测窗口对应的换页增量，不能归因为换页，也不能排除。

## 3. 峰值与真实矩阵形状

脚本：`scripts/diagnose_ane_utilization.py`
证据：`.tmp/ane_utilization_T256.json`

实际模型 dim=256、heads=8、dim_head=64；QKV 权重 [1536,256]，输出投影 [256,512]，FFN [1024,256]/[256,1024]。FMA 按2 FLOPs。以下 execute 为预填输入后同步 API 的墙钟时间，不是硬件 MAC 利用率计数器。

| 算子 | execute median ms | execute TFLOPS | 公共调用 median ms |
|---|---:|---:|---:|
| QKV | 1.528 | 7.91 | 10.003 |
| 输出投影 | 0.746 | 5.40 | 2.242 |
| FFN up | 0.747 | 10.78 | 3.618 |
| FFN down | 1.287 | 6.26 | 3.627 |
| time QK | 3.152 | 1.28 | 18.400 |
| time AV | 4.254 | 0.95 | 9.204 |
| freq QK | 2.827 | 0.33 | 5.817 |
| freq AV | 2.421 | 0.39 | 6.188 |
| dense conv1×1 chain8 | 4.206 | 16.34 | 7.679 |
| dense linear512 chain4 | 2.243 | 14.36 | 4.315 |

softmax(exp2 + @ones) 的 execute 中位数：time 6.493ms / freq 2.437ms；完整 QK-softmax-AV 核心：time 10.197ms / freq 5.096ms。**孤立算子时间不能相加当作融合图归因**。

14项全部编译成功、输出有限、public-call 与 execute 输出逐位一致。计时窗口均没有 Swapins/Swapouts；仅少量 page-in/解压记录。该事实只覆盖这些短测试窗口。

## 4. 候选优化的门禁

研究实现：`scripts/ane_attention_candidates.py`（不由生产路径选择）
测试：`scripts/diagnose_ane_attention_layout.py`、`scripts/diagnose_ane_packed_attention.py`

### batch/head 打包

freq 独立 QK：3.130→1.933ms；AV：2.452→1.468ms，均逐位相同。

但完整 attention：time 14.295→17.062ms，freq 9.061→11.216ms。T64/d2 整图 baseline 20.501ms，仅 freq 打包 21.466ms，全打包 22.204ms。所有数值逐位相同，但性能回退；不集成。尚未进一步分离布局转换与编译融合变化的具体成本。

图内60→64填充版本编译失败，只说明该实现未能执行，不构成“对齐无效”的证据。

### RoPE

直接偶/奇隔元素切片：time 2.324→63.726ms，freq 1.742→68.464ms；否决。

把 Q/K 权重预排为偶数半块/奇数半块，运行时连续切半旋转：
- 完整 time attention：15.411→15.087ms。
- 完整 freq attention：9.281→9.241ms。
- T64/d2 整图：20.241→20.153ms，约0.44%。

各项输出逐位相同；整图 NumPy cosine 均0.99987558。没有足够证据认为存在稳定实用收益，不改默认路径。

## 复现

从 pymss-ane 目录执行，避免并行 GPU/ANE 跑分。原始 JSON 包含逐轮数据。

```sh
.venv/bin/python scripts/diagnose_ane_gpu.py --T 512 --rounds 8
.venv/bin/python scripts/diagnose_ane_gpu_bridge.py --T 512 --rounds 8
.venv/bin/python scripts/diagnose_hybrid_e2e.py --T 64 --depth 2 --rounds 4 --reference
.venv/bin/python scripts/diagnose_ane_utilization.py --T 256 --reps 15
.venv/bin/python scripts/diagnose_ane_attention_layout.py --T 256 --reps 15
.venv/bin/python scripts/diagnose_ane_packed_attention.py --stage block --T 256 --variants baseline,rope_halves
.venv/bin/python scripts/diagnose_ane_packed_attention.py --stage full --T 64 --depth 2 --reference --variants baseline,rope_halves
```

后续若要归因硬件，需另取 GPU 设备时间、频率/功耗及内存流量证据。当前数据不支持以软件接口墙钟差异直接推出硬件机制，也不支持宣布图级优化已被穷尽。
