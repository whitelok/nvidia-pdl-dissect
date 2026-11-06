# NVIDIA PDL 深入分析与 B300 实验记录

日期：2026-11-06。原始 `demo.cu` 保持不变；实验程序见 [pdl_probe.cu](../experiments/pdl_probe.cu)。官方与 CUTLASS 固定版本证据见 [来源核查](pdl-official-research.md)。本报告区分文档保证、源码观察、实验结果与推断。

## 四个问题的结论

1. **会有竞争风险，且本次确实观察到前驱变慢。** 在 64 MiB 权重实验中，PDL 开启后 RMSNorm 的插桩时间跨度从约 1.5 µs 增至 3.6–4.0 µs。但即使不发显式预取指令也会变慢，所以不能把全部差异归因于预取带宽；调度、驻留、内存和插桩都可能参与。端到端收益必须单独评价。全量预取相对不预取明显更慢。
2. **CUTLASS 支持、此次 launch 启用、实际重叠、最终提速是四个不同判断。** v4.3.5 的多个 SM90 GEMM 路径具有 wait/trigger，但 host `launch_with_pdl` 默认 false，编译宏也有限制。不能据此声称所有 sm90+ kernel 都默认开启或都会提速。
3. **可以串联。** 中间 kernel 可以先 wait 前驱、访问依赖数据，再 trigger 后继并继续收尾。本次三 kernel 链计算 1→2→3 正确。允许提前调度不等于保证同时执行，不能用相互忙等依赖并发来保证正确性。
4. **未在所核查官方资料中找到 PDL 专属的 stream 数量上限。** 本次创建并使用 1、2、4、8、16、32、64、128 个 stream 均成功；这不是最大值，也不是同时驻留 kernel 数。普通 stream launch 属性作用于同 stream 前驱，不能自动建立跨 stream 依赖。

## 1. 原始 demo 与文章核对

文章通过用户已登录 Chrome 的现有知乎标签页读取；直接 HTTP 访问返回 403，未将访问失败误认为读过原文。[文章](https://zhuanlan.zhihu.com/p/1993749188043888423)。

### 依赖和调用位置（源码观察）

`x, weight1 → mimic_rmsnorm → y → mimic_matmul → z`，weight2 与 RMSNorm 输出无依赖，理论上可提前加载。

- `mimic_rmsnorm` 在平方和输入循环之后调用 `cudaTriggerProgrammaticLaunchCompletion()`，随后执行归约和 y 写回。
- `mimic_matmul` 先执行 weight2 加载循环，随后调用 `cudaGridDependencySynchronize()`，再读取 y。
- host 构造 `cudaLaunchAttributeProgrammaticStreamSerialization`，但实际使用 `cudaLaunchKernelEx` 的 PDL 分支被注释。**原文件默认运行普通串行 launch。** 仅仅构造属性不会启用 PDL。
- trigger 释放启动条件，不代表 y 写完；wait 保证消费前驱数据的安全。源码这一前后布局符合 PDL 的基本意图，但计算本身有严重问题。

### 阻止将原 demo 当作正确基准的问题

| 位置 | 问题 | 后果/修正 |
|---|---|---|
| norm 平方和循环 | 行内索引又加 `blockIdx.x*blockDim.x`，步长又乘 `gridDim.x` | 默认 K=512、block=512 时只有第 0 行有输入迭代；每行应从 threadIdx.x 开始、按 blockDim.x 跨步 |
| norm shared 归约 | 未将 sum_squared 写入 sdata，也没有归约前初始化屏障 | 读未初始化 shared；应写入后同步 |
| norm 权重读取 | trigger 前没有读取 weight1；写 y 时还会读 x 与 weight1 | 不能假设 trigger 后不再占读带宽 |
| MM 预取 | 每轮覆盖同一个 32×32 s_w | 只剩最后一个 K tile；后续所有 K tile 使用同一份权重，结果错误 |
| MM grid | 列方向使用 hidden_size 而非 output_hidden_size | 默认 16 个列 tile 而非正确的 4 个，发生错位访问及越界 |
| MM 边界 | 无 M/K/N 尾块谓词 | 非整除尺寸越界 |
| main | 输入未初始化、没有数值校验、没有错误检查和计时 | 运行不报错也不能证明正确或更快 |
| 编译示例 | 注释写 code.cu；std::min 未显式包含 algorithm | 修正文件名并显式包含头文件，避免依赖传递 include |

### 预取到底是什么

原代码是普通 global load 写入 shared，并非专门的 L2 prefetch 指令。普通 load 可能填充 cache，但不能把它等同为“所有权重保存在 L2”。每 CTA 每轮写 1024 个 float，即 **4 KiB shared 权重 tile**；K=512 时 16 轮，源码层面发出 64 KiB 权重读取/CTA，却不断覆盖同一 4 KiB buffer。有效计算必须逐 tile 使用正确权重。

原问题的总权重为 K×N×4 = 512×128×4 = **256 KiB**。若修正 grid，4 个 N tile、1 个 M tile 的逻辑读取量覆盖这 256 KiB；M tile 增多时可能重复读取权重。原错误 grid 有 16 个 CTA，不能将其读取量当作合法矩阵的预取量。源码逻辑读取量、实际 cache sector 请求、实际 HBM 流量也不同，后两者需要 profiler 计数器。

文章描述 x/weight1 先进入 shared 再预取 weight2 到 L2；当前 demo 并未这样实现：x 的平方和在寄存器中累积、weight1 在 trigger 后才读取，而 weight2 显式落入 shared。文章的概念性性能图和其所述约 10% 提升，不构成这份有错误 demo 的数值正确性证明，也不保证本实验复现相同比例。文章把 wait 简写成加载同步时，应补充它等待的是前驱 grid，而不是替代后继自身异步 copy 的完成协议。

## 2. 资源竞争、收益边界与处理方法

**文档保证：** PDL 允许机会性重叠，不预留带宽，也不保证 concurrent residency。[CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

**推断：** 预取消耗 load 发射、cache 请求队列、L2 容量和带宽，cache miss 时还消耗 HBM 带宽；后继 CTA 驻留还占寄存器、shared、warp/CTA 槽位。即使数学运算独立，也不表示硬件资源独立。RMSNorm 的尾部写回、再次读取输入和 scale 都可能受影响。若前驱已吃满内存带宽，强预取可能只是抢占；若前驱尾部主要计算且有空闲资源，少量权重/描述符准备更可能被隐藏。

较大总权重不必全量预取：预取窗口应由前驱剩余时长、可用带宽和 cache 容量共同决定，可用 `预取字节 ≲ 可用带宽 × 可重叠时长` 作初始预算，再测量调整。超过 L2 或过早预取还可能在使用前被逐出。不能只靠权重尺寸决定策略。

本实验独立测量：端到端 CUDA event（两 kernel 之间不插 event），以及每个 norm CTA 的 globaltimer 开始/结束，取 `max(end)-min(start)`。后者包含 CTA 调度跨度和插桩成本，不是无扰动的纯算术时间。正的 overlap_window 表示最早 MM CTA 已进入而 norm 仍未结束，不等于预取完成量，也不等于 Tensor Core 已开始消费 y。

工程处理顺序：先修正确性；只提前做真正独立的工作；以有限 tile 或 L2 hint 预取替代覆盖 shared 的伪预取；调节 trigger 位置；测量不同 CTA 资源配置；最后才扩大预取量。原 MM 的 1024 threads/CTA 与本实验的 256 threads/CTA 并不等价，本实验不声称生产 GEMM 性能。

## 3. CUTLASS 为什么集成 PDL

固定 **v4.3.5 / 4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b**。详见 [源码证据表](pdl-official-research.md#4-cutlass-固定源码路径)。

- `arch/grid_dependency_control.h` 包装 PTX 指令并受编译条件控制；SM90 条件涉及 `__CUDA_ARCH_FEAT_SM90_ALL`，不能只看宏名字。
- `gemm/device/gemm_universal_adapter.h` 的 `launch_with_pdl=false` 与 kernel 里的指令是两层开关。
- 普通、ping-pong、cooperative SM90 GEMM 有具体 wait/trigger 路径；普通模板在 mainloop load 前 wait，在 epilogue store 前 trigger，利用的是后继准备与前驱收尾之间的窗口。
- Example 63 使用专用 weight-prefetch kernel，显式控制 prefetch_ratio 和 overlap_ratio；不是每个普通 GEMM 都具备等价预取能力。

**推断：** 对短 kernel 链，消除部分启动/调度空隙即可有可见百分比收益；对于很长的 GEMM，隐藏 1 µs 的百分比很小。软件中广泛保留低层协议接口，是提供组合优化能力的证据，不是每个 shape 的收益承诺。本次未编译运行 CUTLASS benchmark，因此没有 CUTLASS 性能数字。

## 4. 链式协议、内存可见性与 stream 上限

**文档保证：** 所有前驱 CTA 已 trigger 或退出才具备提前启动条件；trigger 不是数据 ready。`cudaGridDependencySynchronize()`/`griddepcontrol.wait` 等待依赖 grid 完成及其内存操作可见。PTX wait 是执行线程的语义；warp specialization 中还需正确的线程间交接屏障。[PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)

本实验 `chain` kernel 的核心是：

```cpp
// A: stage=0; B: stage=1; C: stage=2
if (stage) cudaGridDependencySynchronize();
cudaTriggerProgrammaticLaunchCompletion();
if (threadIdx.x == 0) *out = stage ? *in + 1 : 1;
// A 普通 launch；B、C 的 launch 设置 programmatic serialization 属性。
```

B 既 wait 又 trigger。触发后才写自己的 out 仍然正确，因为 C 在读 B 的 out 前必须 wait B 完成。此例不依赖后继反向通知，串行执行也能终止，因此不要求并发才能保证进度。实际代码对各 stream 分配独立输出，host 同步后检查全部 1、2、3。

| 概念 | 本次可支持的结论 |
|---|---|
| stream 可创建数量 | 没找到统一的 PDL 特定上限；创建受 host/driver 资源及错误返回约束 |
| 多 stream 提交 PDL | 本机测试到 128 个创建成功并提交正确；不是上限探索，也不是全都同时执行 |
| 并发 resident kernel 数量 | 架构、资源、调度等限制，不能由 stream 数推算；concurrentKernels 能力也不是具体并发数量 |
| 依赖链长度 | 可组合连续多节点；未找到所引 PDL API 的固定链长上限；只实测三节点 |
| 跨 stream | 普通 programmatic stream serialization 不自动建立跨 stream 依赖；独立 stream 间需显式依赖机制 |

CUDA Graph 的 programmatic edge/port 与 programmatic event 是额外机制，不能用“普通写法同 stream”推导所有 PDL 场景都禁止跨流/图组合。图依赖需要单独配置和验证，本次没有实测。[CUDA Graph PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#use-in-cuda-graphs)

## 5. 实验环境、方法与完整结果

实验通过用户现有 Chrome 远程 terminal 操作，选物理 GPU 7，`CUDA_VISIBLE_DEVICES=7` 后程序设备号为 0。

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| compute capability / SM | 10.3 / 148 |
| L2 | 132644864 bytes（126.5 MiB） |
| 驱动 / nvcc | 580.159.03 / CUDA 13.0 V13.0.88 |
| Nsight Systems | 2026.4.1.191-264138605071v0 |
| Compute Sanitizer | 2025.3.1.0 build 36400806 |
| 现有任务 | SGLang scheduler；GPU7 显存占用 238770 MiB / 275040 MiB |

用户将准入条件明确放宽为 GPU 利用率 0% 即可。最初各卡均 0%，保留了全部现有进程。一次 large.csv 执行前采到 68%，该轮不作为主要证据。large_idle 和 gated 系列用 0% 条件启动；gated_small 与 gated_large 连续运行，未在二者间再次单独做 0% 准入。测试前后采样并非连续隔离保证，不能排除测试过程中后台服务接到请求。未修改 GPU clocks、功耗或独占模式。

### 程序与正确性

程序是正确的 tiled FP32 GEMM（非 Tensor Core），RMSNorm scale 固定为 1。CPU 使用 double 累积参考，输入为确定性非零有符号数。每次迭代检查有限数及 `abs_error <= 1e-3 + 1e-3*abs(reference)`。两模式相同 kernel、相同数据、相同 hints，只切 launch 属性。PDL 关闭时 kernel 仍保留 wait/trigger，以减少代码差异。

- small: M=32,K=512,N=128，权重 256 KiB，最大绝对误差 1.13248825e-6。
- large: M=32,K=4096,N=4096，权重 64 MiB，最大绝对误差 6.49690628e-6。
- 尾块测试 M=17,K=65,N=33，经 Compute Sanitizer memcheck，`ERROR SUMMARY: 0 errors`；不是 racecheck 结论。
- 每个配置 5 次预热，正式 gated 系列 30 次。奇偶轮交替 PDL 开关的测试次序。
- early=0：平方和读取后触发；early=1：norm CTA 入口触发。
- prefetch_rows=0/16/K：在 wait 前对相应权重前缀发 `prefetch.global.L2` hint，仅第一个 M tile 发出，避免 M 维重复。每线程每行发 hint，存在同 cache line 冗余请求；这是受控微基准，不是最优预取实现。hint 不承诺实际完成量。
- large 的 16 行前缀是 256 KiB，K 行是 64 MiB；本次 64 MiB 仍小于 L2，**没有验证大于 L2 的权重**。预热后 cache 状态接近重复使用，未做 cold-cache 扫描。

### 测量迭代的必要修正

初版 ee032e4 在 host 连续提交 norm/MM，但 1–1.5 µs 的 norm 先完成，未观察到重叠。small 端到端约 25 µs、large_idle 约 151 µs。初版数据保留，不能据它断言 PDL 无效。

修正版 0b5e51c 在 event 之前加 1 ms 单线程 gate kernel，给 host 预先排队的机会。gate 不计入 pair 的 event 时间；但它可能改变 GPU 活跃状态，因此这些数字表示该控制条件下的 kernel pair latency，不是完整服务请求耗时。它也不能无条件保证 host 永不被抢占；Nsight 时间线用于交叉验证。

### gated 数据（全部配置）

以下时间单位均为 µs。E2E 为中位数，SD 为样本标准差；Norm 与 overlap 为中位数。每行 n=30。数据从远程统计输出逐项抄录到 [summary.csv](../results/summary.csv)。

| shape | PDL | prefetch rows | early | E2E | SD | Norm | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 0 | 0 | 0 | 25.968 | 0.429 | 1.168 | 0 |
| small | 0 | 0 | 1 | 26.400 | 0.407 | 1.184 | 0 |
| small | 0 | 16 | 0 | 25.600 | 0.377 | 1.184 | 0 |
| small | 0 | 16 | 1 | 25.600 | 0.408 | 1.152 | 0 |
| small | 0 | 512 | 0 | 25.600 | 0.351 | 1.168 | 0 |
| small | 0 | 512 | 1 | 25.600 | 0.318 | 1.168 | 0 |
| small | 1 | 0 | 0 | 23.552 | 0.584 | 1.312 | 0 |
| small | 1 | 0 | 1 | 22.368 | 0.533 | 1.536 | 0.672 |
| small | 1 | 16 | 0 | 23.552 | 0.392 | 1.424 | 0.128 |
| small | 1 | 16 | 1 | 23.552 | 0.576 | 1.536 | 0.624 |
| small | 1 | 512 | 0 | 23.552 | 0.509 | 1.568 | 0.256 |
| small | 1 | 512 | 1 | 23.552 | 0.516 | 1.536 | 0.688 |
| large | 0 | 0 | 0 | 159.104 | 7.428 | 1.504 | 0 |
| large | 0 | 0 | 1 | 158.256 | 7.856 | 1.504 | 0 |
| large | 0 | 16 | 0 | 157.728 | 7.304 | 1.488 | 0 |
| large | 0 | 16 | 1 | 157.616 | 7.501 | 1.504 | 0 |
| large | 0 | 4096 | 0 | 172.016 | 1.807 | 1.504 | 0 |
| large | 0 | 4096 | 1 | 171.424 | 1.605 | 1.504 | 0 |
| large | 1 | 0 | 0 | 157.824 | 6.865 | 3.568 | 2.192 |
| large | 1 | 0 | 1 | 157.696 | 6.987 | 3.840 | 2.976 |
| large | 1 | 16 | 0 | 157.840 | 7.134 | 3.696 | 2.432 |
| large | 1 | 16 | 1 | 157.712 | 6.926 | 3.840 | 2.944 |
| large | 1 | 4096 | 0 | 169.488 | 2.778 | 3.952 | 2.688 |
| large | 1 | 4096 | 1 | 171.008 | 4.800 | 4.032 | 3.232 |

### 如何解释这些数字

**实验结果：** small、early=0、pre=16：PDL 将 E2E 25.600 降至 23.552 µs（约 8.0%），但 Norm 从 1.184 增至 1.424 µs。前驱变慢与整个链变快可以同时成立。small、early=1、无 hint 的 26.400→22.368 µs 约 15.3%，仅适用于该微基准，不能替文章背书。

large、early=0、pre=0：159.104→157.824 µs 仅约 0.8%，而 SD 约 7 µs；不能宣称这一点差异为稳定显著加速。Norm 1.504→3.568 µs 的变化明显大于 small。pre=4096 时 Norm 达 3.952 µs、E2E 达 169.488 µs；与 PDL+无预取的 157.824 相比反而慢约 7.4%。关闭 PDL 的全量预取也慢：159.104→172.016 µs。**预取额外工作的成本本身就可能超过 cache 收益。**

**推断边界：** 这些结果支持“重叠可能拖慢前驱”和“多预取不一定更好”，不能证明某个特定硬件瓶颈。即使 pre=0，MM 的启动、wait、计时和驻留也与 norm 重叠；pre=K 相对 pre=0 的差异才更接近预取新增影响。没有采集 HBM/L2 吞吐、hit rate、warp stall 等计数器，因此尚不能量化带宽竞争与资源驻留各占多少。

### 时间线和链正确性

`gated_trace.nsys-rep` 已成功生成并导出 SQLite，共处理 5527 events。按 kernel 名筛选、同序配对 norm/MM，72 对（包含 warmup），35 对有时间交集，最大 1504 ns。此结果独立于 globaltimer 插桩，证实实际存在重叠；profiling 的性能数字不混入上表。

同一 trace 有 1530 个 chain kernel，等于 `(1+2+4+8+16+32+64+128) × 2 modes × 3 stages`。所有模式均通过 1→2→3 的检查。完整 kernel start/end/streamId 已导出 `timeline.csv`。此次尚未完成按 stream 与 launch 模式逐条展示 A/B/C 的相邻重叠统计，因此**没有声称三 kernel 同时驻留**。本例的链 kernel 很短且 host 提交未加链专用 gate，不能以其时间线是否重叠否定链式协议支持。

## 6. 命令、原始文件与复现

所有远程实验文件保存在 `/tmp/pdl-dissect-20261106/`。这是远程容器路径，不是本地文件。临时目录随容器销毁可能消失，应在回收 pod 前归档。远程另有 `raw-repo/results/raw/` 副本和 commit `d437c28`；此前尝试远程 push 因没有 GitHub 凭据失败，**未上传**。用户随后明确远程不能上传，后续仅由本地推送源码与报告。没有复制凭据到容器。

本地仓库保留原始 demo、完整实验程序、统计脚本和全配置汇总；逐次 CSV 与 profiler 二进制仍在远程，不能把 summary.csv 当作逐次原始数据。

| 远程文件 | 来源/用途 |
|---|---|
| small.csv/.log | 初版，20 repeats，未消除 host 提交间隙 |
| large.csv/.log | 初版，20 repeats，测试前 68% 利用率，非主要证据 |
| large_idle.csv/.log | 初版，0% 条件启动，30 repeats |
| gated_small.csv/.log | 修正版，30 repeats，主要 small 数据，360 条测量 |
| gated_large.csv/.log | 修正版，30 repeats，主要 large 数据，360 条测量 |
| sanitizer.csv/.log | 初版尾块正确性与 memcheck，summary 在 stdout 文件中 |
| trace.nsys-rep、trace.csv/.log | 初版 profiling |
| gated_trace.nsys-rep、gated_trace.sqlite、gated_trace.csv/.log | 修正版 profiling |
| timeline.csv | SQLite 导出的所有 kernel start/end/stream/name |

### 已使用的主要命令

```bash
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.used,utilization.gpu --format=csv
nvcc --version
command -v nsys
command -v compute-sanitizer

mkdir -p /tmp/pdl-dissect-20261106
cd /tmp/pdl-dissect-20261106
# 初版使用 ee032e4；最终测量使用 0b5e51c。
curl -fL https://raw.githubusercontent.com/whitelok/nvidia-pdl-dissect/0b5e51c/experiments/pdl_probe.cu -o pdl_probe.cu
nvcc -O3 -lineinfo -arch=sm_103 pdl_probe.cu -o pdl_probe

if [ "$(nvidia-smi -i 7 --query-gpu=utilization.gpu --format=csv,noheader,nounits)" = 0 ]; then
  CUDA_VISIBLE_DEVICES=7 ./pdl_probe 32 512 128 30 > gated_small.csv 2> gated_small.log
  CUDA_VISIBLE_DEVICES=7 ./pdl_probe 32 4096 4096 30 > gated_large.csv 2> gated_large.log
fi
# 本次 memcheck 实际对初版二进制执行；最终版只增加前置 gate。
CUDA_VISIBLE_DEVICES=7 compute-sanitizer --tool memcheck ./pdl_probe 17 65 33 1 > sanitizer.csv 2> sanitizer.log
CUDA_VISIBLE_DEVICES=7 nsys profile --trace=cuda --sample=none --cpuctxsw=none -o gated_trace ./pdl_probe 32 512 128 1 > gated_trace.csv 2> gated_trace.log
nsys export --type sqlite --output gated_trace.sqlite gated_trace.nsys-rep
python3 experiments/summarize.py gated_small.csv gated_large.csv
```

最后一条需要将仓库统计脚本放在执行目录相应位置；本次远程使用逻辑相同的 Python 标准库 heredoc 做统计。新复现应给每一轮独立做空闲检查，并保留检查 stdout、命令退出码与连续监控，避免把上一轮 GPU 活跃采样误认为后台活动或反过来。

时间线导出查询：

```sql
SELECT k.start, k.end, k.streamId, s.value
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.demangledName
ORDER BY k.start;
```

## 7. 对 demo 的具体建议与未验证事项

1. 先修正索引、归约初始化、K tile 权重使用、grid 和边界谓词，再初始化输入并做 CPU/cuBLAS 对照。不要直接拿当前 demo 计时作结论。
2. 将 PDL host 开关显式参数化；所有消费者在读取 y 前 wait，所有生产者 CTA 都有合法 trigger/退出路径。不要把 wait 当成异步 copy 完成屏障。
3. 将“预取策略”和“PDL 开关”分开。至少比较串行/PDL × 无预取/有限前缀/全量前缀。本实验表明全量策略会退化，初始优先尝试少量即将消费的 tile。
4. 对生产 GEMM 使用已验证的 CUTLASS 模板或 Example 63 思路；评估 occupancy、寄存器、shared、持久 CTA 调度与前驱的配合，不能把这个 naive GEMM 的比例照搬。
5. 同时报 E2E 和前驱跨度，不在 norm/MM 之间插普通计时 event 改变依赖。用 Nsight 验证提交间隙与实际重叠；最终应补无插桩版本、CUDA Graph 或批量排队的独立复测。
6. 待补：大于 L2 的权重与 cold-cache；更多 M/K/N 与 block/资源配置；连续背景负载监控；Nsight Compute 计数器；真实 CUTLASS GEMM 对照；非单位 norm scale；更长链和链专用时间线；多 stream 吞吐/延迟统计与更大 stream 数探测；跨 stream/Graph programmatic 依赖验证。

本报告的实验范围是可复现的 B300 微基准与源码语义核对。尚未证明某种方案在真实模型、不同架构或不同后台服务负载下都获益。
