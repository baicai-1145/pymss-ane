# ANE attention 融合调查：边界实测与可观测性

本轮研究生产模型形状下的 attention 融合，不以单 program、前端节点数或 E5 TD 块数替代物理融合证据。该轮没有生产优化集成、没有 commit。

团队只读协作报告：`/Volumes/2T/repos/mss/.pi/herdr-team/dqbf3a/reports/`。主报告以修订后的 fusion-audit-evidence-map.md、fusion-design-task2-c3v3-final.md 与后续勘误为准；早期报告的过度归因已撤回。

## 当前结论

- 完整 core 比 QK 切图更快；显式 score buffer 复制不是全部差额。
- 强制导出 score 端口会改变执行耗时；小输出端口影响接近短窗波动。
- 本机编译产物确认 profiling 请求被接受，但这条第三方 e5rt 客户端路径最终 statsMask=0，未取得 per-TD 硬件统计。
- Q 查询行分块表达可以编译且数值正确，但在实际 batch=60 形状反而更慢，不集成。
- **尚未直接观测 score 是否完整写入外存、各块是否驻留片上、具体哪些物理指令融合。不能宣称这些问题已解决或硬件做不到融合。**
- **所有已探通道（时间差、端口导出、profiling 统计、编译器 IR、容器 hook）均已验证到各自边界；继续堆叠同类时间差实验无法回答片上驻留问题。**

## 1. 完整 core 与显式 QK 边界

脚本：`scripts/diagnose_ane_fusion_boundary.py`
证据：`.tmp/ane_fusion_boundary_T256.json`

head=8、dim_head=64；两边均计算 `score=(Q@Kᵀ)/sqrt(D)`、`e=exp(score-max(score))`、`out=(e@V)/rowsum(e)`。拆图版把 QK/scale 与后续操作放入两个 program，使用 fp16 持久缓冲区，显式 np.copyto 交接 score。所有输入预填；15轮随机配对；直接逐段计时而非相减独立基线。

| 方向 | 完整 median ms | 拆图 median ms | score复制 median ms | 配对计算段差 median ms |
|---|---:|---:|---:|---:|
| time，B60/S256 | 10.658 | 13.715 | 1.618 | 1.260 |
| freq，B256/S60 | 5.744 | 7.930 | 0.376 | 1.741 |

配对计算段差是每轮先算 `qk_execute + tail_execute - full_execute` 再统计，不能拿上述中位数直接相加求闭合。显式交接 score 分别为62.91/14.75MB。

两方案输出逐位一致；NumPy fp64 首lane cosine >0.9999999；计时窗口无swap进出。差额仍混有多一次调用、端口/布局和编译优化边界，不等于 SRAM 驻留收益。

编译结构指纹：time full4个解析器TD块，qk/tail分别3/3；freq full5，qk/tail分别5/4。**TD块数不等于物理内核数、边界数或可线性计价的成本单位。** 原始JSON记录各程序 build_dir、MIL SHA、E5候选文件与签名。

## 2. M4 profiling 请求与统计是否到达

脚本：`scripts/diagnose_ane_profile_channel.py`
证据：`.tmp/ane_profile_channel/{summary.json,ane_events.ndjson,control/,profile/}`

两个独立子进程：环境不设 ANEFORGE_PROFILE 与设为1；各自新 build_dir，同一个小 core。无sudo，log stream开启signpost；不把仪器化耗时拿来跑分。

观测：
- 两份 model.mil 字节相同。
- E5同为3040B；原始 build-info 中 `built-for-profiling` 的值 false→true（0x164长度05→04，0x168起字符串变化），证明编译请求生效。现有 E5 解析器的 build_info 未完整列出该属性，因此此条依据原始字节核对，不谎称解析器已解析到。
- 两次编译、加载、执行均成功，输出有限。
- control/profile客户端PID9751/9776日志均 `model.perfStatsMask:0`；aned PID977两次均记录 `ThirdPartyAppUsingANE StatsMask=0` 和加载行 `statsMask=0`。
- 402条日志中140条signpost；实际statsMask相关记录为6条。初版正则误把 senderProgramCounter（代码地址）当统计字段，已修复并离线重解析，原始日志保留且记录SHA。
- cachedModel.length为81920/98304，仅登记差异，不据此推断具体插桩机制。

结论限于本机这条第三方路径这次请求未取得逐TD统计。不能沿用其他芯片“profiling导致拒载”的描述：本机两模式均加载成功；也不能推出硬件无计数器或所有路径都不可用。

## 3. 强制输出中间 score 的扰动

脚本：`scripts/diagnose_ane_fusion_ports.py`
主证据：`.tmp/ane_fusion_ports_T256_warm.json`
首轮保留：`.tmp/ane_fusion_ports_T256.json`

为避免顶层编译路径不同，使用原始 lowering 的 `compile(opt=0)` 与内部 `aneforge._compile.compile_multi`。后者不是当前安装版顶层导出API。

6种变体：mono_raw、multi_single、multi_tiny_port、multi_score_port、identity_tiny、identity_score。完整集编译后统一预热3遍，随后20轮随机交错；预填输入、只execute、不在计时区读回输出。所有主输出与identity门禁逐位一致。保持全部样本，不根据观察后的阈值删离群。

| 配对比较 | time中位差 ms | 正/负组数 | freq中位差 ms | 正/负组数 |
|---|---:|---:|---:|---:|
| multi_single − mono_raw | +0.075 | 见JSON | +0.016 | 见JSON |
| tiny_port − multi_single | −0.039 | 8/12 | +0.030 | 11/9 |
| score_port − multi_single | +0.442 | 20/0 | +0.914 | 19/1 |

identity_tiny中位约0.135/0.148ms；identity_score约1.340/1.590ms。identity是包含编译优化、端口布局与数据处理的组合负载，不是纯固定调用成本或已知字节的DMA微基准。

计时窗没有swap；time压缩1577页/解压411页，freq解压890页。仅有整窗计数，不能把分页变化归到某一轮；该短窗结果不保证跨进程、跨热态稳定。

结论：强制导出score有可见的总影响，tiny输出在本窗口影响接近波动。增加输出还会改变lowering、分配、调度；Δport不能直接定价DRAM流量，也不能判定原图是否驻留。小端口与score端口都可能改变TD指纹，且指纹块数与耗时不单调。一个目录里有多个E5文件时仅作为候选集合登记，未唯一定位活动产物。

## 4. 查询行分块候选

脚本：`scripts/diagnose_ane_query_tiles.py`
证据：`.tmp/ane_query_tiles_S256.json`

S256/H8/D64；Q行切成64或128行，每块仍沿完整K维归约，固定每块div、`@ones(S,1)`，最后沿Q轴concat。输入种子相同，opt=0，无RoPE/投影。每个batch的全部程序编译完成后统一预热，再15轮随机交错。

| batch | baseline median ms | Q64 median ms | Q128 median ms |
|---|---:|---:|---:|
| 1 | 0.263 | 0.222 | 0.243 |
| 8 | 1.421 | 1.521 | 1.455 |
| 60（模型time方向） | 10.245 | 11.355 | 10.711 |

9个图全部编译成功、输出有限且相对baseline逐位相同，NumPy首lane门禁通过。每case记录MIL SHA及1个E5候选（JSON字段为e5_candidates），解析器TD块数9例均为4；不能将相同TD块数解释为相同物理执行。B60/Q64全部15轮配对更慢；Q128仅3/15轮更快。B1的轻微收益不能外推生产形状。

B1/B8无压缩或解压；B60整窗压缩1660/解压317页；全部无swap。没有剔除样本。

判定：本Q切片+concat表达在生产batch未获益，不集成。前端节点数增加只说明表达变化，不证明设备插入了多少搬运。最大的单个score节点变小不等于峰值活跃内存变小；native调度可能不同。本结果不反证片上分块融合的存在性或可行性。

## 5. 编译器 IR 通道验证（后续轮次）

脚本：`scripts/diagnose_ane_compiler_ir.py`
证据：`.tmp/ane_compiler_ir/summary.json`

对照（不设变量）与 dump（`ESPRESSO_PRIVATE_ANE_DUMP_IN_MEMORY_IR=1`）两个子进程各自编译并执行同一最小 core，编译前后扫描 scratch／home／tmp 目录，同时监听 ANECompilerService 日志。结果：两模式均成功编译执行，均无新增 `anecir.*` 或 `net.plist`；日志无 anecir 相关事件。`/tmp/anec_out*`、`/tmp/ane_conv_case` 下的 net.plist 为历史手工实验产物，不属于本通道。

结论：该环境变量在本机无 sudo 第三方路径未触发 IR 转储，与“编译工作在 root XPC 服务内完成、客户端环境变量不可达”的结构观察一致。未触发不证明通道不存在；若未来取得 root 或编译宿主内执行能力，这是融合单元级证据（哪些算子落进同一编译单元）的首选通道。

## 6. 证据审查更正

- 旧b22bcfa缓存是39输入T512全模型，不能用84-op PM2日志命名它。
- `optprobe/net.plist`是手工构造的netlist输入，不是编译器IR dump。
- surf_hook已执行但0命中；有工具源码不等于成功捕获容器。既有生产hwx来源与该hook无关。
- 本机片上容量未在本轮测出，不把其他ANE代际的2MB数据当M4事实。
- 两个shape的中位差不足以识别一个多未知数因果模型；不把TD块数兑换每刀成本，不把identity与额外端口当可独立相加的硬件分量。

生产 `scripts/build_graph.py` 无本轮修改。研究脚本与原始样本保留供复现；当前已验证的模型路径不变。
