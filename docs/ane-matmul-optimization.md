# ANE matmul/bmm 优化调查：成本模型与已排除路径

## 集成变更（2026-10，native softmax）

`build_graph.py` attn() 单块路径默认改用原生 `softmax(axis=-1)`（cfg `attn_native_softmax=False` 可回退；分块路径保留手工链）。依据：
- 单元 A/B：+7.3–7.9%（`diagnose_ane_native_softmax.py`，`.tmp/ane_native_softmax.json`）
- 整模型 T=256 d12：359.9→348.3ms（+3.3%），节点 2662→2494
- parity 门禁（T=16 d12，fp64 参考）：cos 0.999986 持平，rel_med 0.0109 略优于 manual 0.0111
- 来源：maderix/ANE 训练管线同样用原生 softmax 算子（mil_dynamic.h:194）；task-21 只证明了 Espresso 会分解它，未比较与手工链的速度——实测编译器自己的分解调度更省（-168 节点）

动机：maderix/ANE 项目报告 conv1×1 表达 3×、int8 1.88× 加速。逐项在我们的生产形状上验证，全部不迁移；换来的是首份 M4 ANE matmul 成本模型。

## 成本模型（B=60 S=512, opt=0, 稳态）

```text
matmul 延迟 ≈ setup(K) + stream(N)·N
  K=64:  0.187 + 0.000998·N ms    (R²=1.000)
  K=512: 0.774 + 0.001657·N ms    (R²=0.998)

bmm（激活×激活）：
  QKᵀ D=64:  1.32T / 31GB/s      （生产形状 60,8,256,64）
  D=128 内维: 1.87T（1.6×/位宽）  ← 但模型 D=64 固定，不可用
  AV N=64:   0.94T / 22GB/s       ← bmm 结构性，见下
```

- setup 随 K 增长（0.19→0.77ms，8×K→4.1×），是每 op 调度/权重面设置成本；
- **权重跨 execute 驻留**：稳态延迟平坦（25次执行尾部 3.40ms 无漂移），排除每次重流 DRAM；
- N=256 的 6.7T"慢"纯粹是 setup 摊销，不是带宽或 tile 悬崖（N 256→1536 线性拟合 R²=0.998）。

## 已排除路径（全部有数值门禁+交错计时）

| 路径 | 结果 | 原因 |
|---|---|---|
| conv1×1 表达投影 | **慢15–35×**（QKV 1.96→33.3ms） | 我们的序列形状（C=256,S=512空间）对conv分块最差；maderix的3×来自视觉形状（C=512,64×64空间）。数值逐位一致，纯执行路径差异 |
| int8 权重（W8A16） | **0.96–1.02×** | ANE 上无速度收益，还引入 rel_l2≈0.7% 误差。仅省内存时用 |
| N 补零到 512/1024 | 有效吞吐反降至 4.0T | 补的 flop 全额付费，摊销救不回 |
| split-K（K=256切块累加） | 慢 1.5–2.5× | 每块再付一次 setup；K=384/512/768/1024 全体变慢 |
| head 打包 bmm | 单算子快、完整块慢（历史轮次） | 融合边界变化吃掉收益 |
| conv1×1 变体 big_linear | 早已评估 | transposes 成本超过 conv 收益（build_graph.py 注释在案） |

## 生产投影现状 vs 峰值

| 算子 | K | N | 实测 | conv链峰值参照 |
|---|---:|---:|---:|---|
| QKV | 256 | 1536 | 12.7T | 16.3T（78%） |
| FFN up | 256 | 1024 | 11.1T | （68%） |
| FFN dn | 1024 | 256 | 6.4–7.2T | setup摊销，无解 |
| out proj | 512 | 256 | 6.0–6.8T | 同上 |

## 剩余结构性瓶颈（带数量级）

1. **AV 的 N=64**：每头输出 64 通道是模型结构（8×64=512），bmm 逐头 N=64 → 0.94T。这是 22GB/s 流送下的形状界，非带宽界（M4 DRAM ~120GB/s）；
2. **QKᵀ 内维 D=64**：1.32T；D=128 可到 1.87T 但 D 是模型超参；
3. **attention 链 ~7 个小 op 的 setup 税** ≈ 12层×3ms ≈ 全模型 3%——量级不足以成为主攻方向；
4. attention 整体 ≈ 模型时间 10%，投影已 68–78% 峰值。

## 结论

全 ANE 单 program 路径在生产形状上已系统性逼近该表达方式的可行上限：
- 投影层：11–14.7T（68–78% conv 峰值），所有已知替代表达（conv/int8/pad/split）实测更差；
- attention：瓶颈是 bmm 形状界（D=64 内维、N=64 输出），源于模型超参而非构图；
- 进一步收益需要改变问题本身：更大 batch 摊销 setup、模型结构（D=128 头）、或等待编译器/硬件路径变化（SDPA 硬件单元修复）。

证据：`.tmp/ane_matmul_vs_conv.json`、`ane_int8_proj.json`、`ane_pad_n.json`、`ane_split_k.json`、`ane_n_sweep.json`、`.tmp/ane_n_sweep_k64/`、`.tmp/ane_bmm_d/`、warmup trend（weight residency）。
脚本：`scripts/diagnose_ane_{matmul_vs_conv,int8_weights,pad_n,split_k,n_sweep}.py`

## ANEC 编译器选项扫描（ANE_ANEC_FLAGS 通道，T=256 全模型）

通道：`ANE_ANEC_FLAGS` env → `e5rt_e5_compiler_options_set_custom_ane_compiler_options`（aneforge dispatch 源码 task-20 钩子）。选项字典来源：ANETools/ANECompiler 的 ANEC class-dump（60+ 旋钮）。

| 选项 | 结果 | 判定 |
|---|---|---|
| baseline | 348.5ms | — |
| SpatialSplit=Memory | 351.6ms (+0.9%) | 调度改变但更慢；输出逐位一致 |
| InputAndConstantCaching=true | 348.4ms | 默认已开 |
| L2CacheMode=L2EnableResident | 348.7ms | 无效 |
| EnableKernelRewind=true | 348.8ms | 无效 |
| FoldScale=true | 348.9ms | 无效 |
| MemCacheStrategy=Simple | 348.6ms | 无效 |
| DisableContextSwitching=true | 349.1ms | 无效 |
| ScanWeightsForCompression=true | 348.1ms | 无效（权重已稠密打包） |
| DramTensorPriorityType=BestFitReuse | **编译被拒** | 选项集已漂移（2022 dump vs 26A434） |

结论：8个有效选项全部落在±1%噪声带内；唯一改变行为的选项（SpatialSplit）变慢。**ANEC默认调度对该图形状已近其能力上限，编译器旋钮通道挖尽。** 附带产出：in-memory路径optionsPlist在26A434要求plist字节且实际不透传到ANEC（无效值静默通过、DisableOptimizations不改编译时间）；Path-A生产模型直连探针（.tmp/prod_opts_probe.m，T=16基线19.55ms）。

## D=128头反事实（重训决策数据，diagnose_ane_d128_heads.py）

数学事实：训好的8×64头不能免重训重排成4×128（伪头会让q跨原头与k内积，函数改变）。此测为**重训反事实**：同hd=512改H=4/D=128的attention核心。

| 形状 | H8/D64（现状） | H4/D128（重训后） | 加速 |
|---|---:|---:|---|
| 核心T=256 | 9.75ms | 6.00ms | **1.63×** |
| 核心T=512 | 37.9ms | 19.9ms | **1.90×** |

全模型外推：T256 348.5→303ms（1.15×），T512 ~1150→~934ms（**1.23×**）；对MLX GPU（1313ms）从1.14×扩到**1.43×**。收益源头即bmm内维64→128的1.6×效率（前轮成本模型），T512时score矩阵bytes也减半（H=4 vs 8），故长序列收益更大（1.90×>1.63×）。数值门禁cos>0.9999全过。**决策归训练侧：需以dim_head=128、heads=4重训mel-roformer，非构图层可达。**

## 音频batch反事实（diagnose_ane_audio_batch.py）

多音频段折叠进带batch维（60→60N）：

| op | B60→240每clip | B480 | 判定 |
|---|---|---|---|
| attention核心T256 | 9.21→9.28ms | 9.67ms（-5%） | 线性，无摊销 |
| attention核心T512 | 39.2→38.4ms | — | +2%边缘 |
| QKV投影T512 | 1.97→2.09ms | — | -6%更差 |

原因：整图单program架构下，每op的setup成本已在编译期摊平，execute时的固定开销只剩dispatch地板（~0.2ms），对9–40ms的计算可忽略——batch只是线性放大stream部分。**加batch对单clip延迟无收益（大batch反而-5%），仅对吞吐型多clip服务有意义。**

## ANE功耗遥测调查（2026-10-26, M4 Mac16,12, macOS 26A434）

powermetrics `ane_power`采样器5轮测量（含满载窗口，T=256全模型350ms/次连续35次）全部返回 **`ANE Power: 0 mW`**（合流/solo/extra-info/三合一均如此）。同窗口CPU 6.4-7.4W、GPU 69-85mW正常上报。结论：**此机此OS的ANE专用轨遥测按0上报（禁用或不存在该轨的读数路径）**，powermetrics无法给出ANE瓦特数。

IOReport免root路径同日证伪：ANE驱动（H11ANEIn，ANE v5代）的IOReportLegend只有中断统计；MCC节点有"ANE DCS RD/WR"（ANE的DRAM流量计数器，IOReportLegendPublic=Yes）但IOReportCreateSubscription私有API参数形态无法确定（5种形态全crash/null，无公开头文件）。全调查档案：/tmp/ane_power_README.md。

能效口径建议：J/推理需瓦特数，此机不可直接测。备选：(a) Instruments Energy Log GUI（无root，含ANE能耗曲线）；(b) 每进程能耗冲击分数（--show-process-energy，无量纲）；(c) 依maderix实测外推：M4 ANE峰值19T@~2.8W，我们353ms/次×77%投影占比≈2.2W满载假设，J/推理≈0.78J（推导值，非实测）。Apple自标powermetrics功耗为"估算值，勿跨设备比较"。
