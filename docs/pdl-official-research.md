# NVIDIA PDL：官方语义与 CUTLASS 固定版本证据

核查日期：2026-11-06。源码基准为 CUTLASS **v4.3.5 / `4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b`**，不是“当前 main”。[GitHub release](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.5) 的 commit 链接给出上述完整 SHA；通过 HTTPS 逐个读取该 SHA 下的源码进行了交叉检查。

## 1. 三个必须分开的判断

| 层次 | 成立需要什么 | 不能据此推出什么 |
|---|---|---|
| 支持 PDL | 设备/工具链满足要求；kernel 具有正确的 wait、trigger 路径 | 默认开启 PDL |
| 启用 PDL | 此次 launch 设置 programmatic stream serialization，且编译出的 kernel 保留正确同步指令 | 两个 kernel 一定重叠 |
| 实际重叠、实际收益 | 调度及资源允许，独立工作或启动开销得到隐藏；需测量 | 任意 shape、任意 kernel 都提速 |

CUDA 将重叠执行定义为机会性行为。不能把需要两个 kernel 同时驻留才能退出的互相等待写成正确性条件，否则可能死锁。硬件支持始于 compute capability 9.0；PTX 指令要求 `sm_90` 或以上。[CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)、[PTX griddepcontrol](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)

## 2. CUDA / PTX 的两种事件

`cudaTriggerProgrammaticLaunchCompletion()` 对应释放后继启动机会。所有 CTA 都执行 trigger 或退出以后，后继才具备提前调度条件。PTX 的 `griddepcontrol.launch_dependents` 在同一 CTA 内重复执行不会增加效果。它不是“输出全部写完”的标记。上游不显式 trigger 时，所有 block 退出会隐式满足触发条件，因此通常失去提前启动窗口。

`cudaGridDependencySynchronize()` / `griddepcontrol.wait` 等待运行时定义的前驱 grid 完成，并保证这些前驱的内存操作对当前 grid 可见。它不是 `__syncthreads()`；PTX 语义针对执行该指令的线程。涉及其他线程消费时，还应审查 kernel 原有线程间交接与同步，不应随意删除其他 barrier。

所以典型顺序是：上游 trigger 后继续收尾；下游先做不依赖上游输出的准备，wait 后才能读取上游结果。提前 trigger 不允许下游直接消费未完成结果。[CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)、[PTX griddepcontrol](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)

最新 PTX 文档另规定，满足 scope/domain 条件的 release fence 可与后继启动建立同步；这不等于裸 trigger 自动完成所有数据发布。本文按完整 grid wait 的标准模式解释 CUTLASS，不把细粒度发布协议混入普通用法。

## 3. stream、链与并发数量

普通 stream PDL 的 launch 属性描述的是**当前 kernel 与同 stream 前一个 kernel**的关系。可以形成 `A → B → C → …`：B 同时等待 A、释放 C；C 同样按协议等待 B。不存在“必须两两独立配对、每一对之外就不能继续”的 API 要求。链式可组合也得到 CUTLASS 文档中“等待前驱并通知后继”的设计支持。[Runtime 属性](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html)、[CUTLASS PDL 文档](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/media/docs/cpp/dependent_kernel_launch.md)

**链长不等于同时执行数量。** 上述官方 API/PTX 资料没有给出“一个 stream 永远最多两个 PDL kernel 同时执行”的通用保证，也没有保证任意多 kernel 同时驻留。不要从双 kernel 示意图推导硬件队列深度；也不要从可链接任意多个节点推导任意深度并发。资源、grid 形状、调度和 wait/trigger 位置影响实际时间线，这一点必须通过目标 GPU 测量。

不要把 stream serialization 属性理解为自动连通不同 stream。CUDA Graph 可使用 `cudaGraphDependencyTypeProgrammatic` 配合 `cudaGraphKernelNodePortProgrammatic` 或 `cudaGraphKernelNodePortLaunchCompletion` 显式表达依赖；官方也有 programmatic event 属性。因此“普通写法是同 stream”与“整个机制只可能存在同 stream 依赖”是不同命题。[CUDA Graph PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#use-in-cuda-graphs)、[Driver execution API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)

## 4. CUTLASS 固定源码路径

下表全部绑定同一 SHA，行号来自该 SHA 的原始文件读取。

| 作用 | 位置与证据 |
|---|---|
| 编译开关与 PTX 包装 | [grid_dependency_control.h](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/arch/grid_dependency_control.h)：`CUTLASS_ENABLE_GDC_FOR_SM90` / `CUTLASS_ENABLE_GDC_FOR_SM100` 联合 CUDA、架构 feature 与 barrier 条件产生 `CUTLASS_GDC_ENABLED`。未开启时包装函数为空。`launch_dependent_grids()` 发出 launch_dependents；`wait_on_dependent_grids()` 发出 wait。`IsGdcGloballyEnabled` 可反映编译结果。 |
| Host GEMM 开关 | [gemm_universal_adapter.h:375](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/gemm/device/gemm_universal_adapter.h#L375)：`run(..., bool launch_with_pdl = false)`；参数传到 kernel/cluster launcher。第 424 行开始的自定义 CUDA adapter 分支拒绝与 PDL 同用。 |
| 真正设置 CUDA 属性 | [cluster_launch.hpp:198](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/cluster_launch.hpp#L198)：构造 `cudaLaunchAttributeProgrammaticStreamSerialization`，值为 1；`numAttrs` 根据 `launch_with_pdl` 决定是否包含该属性。普通 kernel 路径另见 [kernel_launch.h](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/kernel_launch.h)。 |
| 非 persistent 代表 | [sm90_gemm_tma_warpspecialized.hpp:434](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp#L434)：producer 在 mainloop load 前 wait；第 491 行在 epilogue store 前 trigger。可见 trigger 明确早于输出写完。 |
| Ping-pong 代表 | [sm90_gemm_tma_warpspecialized_pingpong.hpp:625](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L625)：多个 producer 路径在相关 load 前 wait；第 810、876 行有不同条件路径的 trigger，不能仅截一个调用就概括所有调度路径。 |
| Cooperative 代表 | [sm90_gemm_tma_warpspecialized_cooperative.hpp:602](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L602)：mainloop producer wait；另有 scheduler/epilogue 路径 wait，第 812 行 trigger。 |

上表证明具体模板路径具备集成点，**不证明全部 CUTLASS kernel 都支持 PDL**。尤其不能只看 GPU 属于 Hopper/Blackwell，就跳过实际模板、编译宏、生成代码及 host launch 参数的检查。该版本 SM90 编译条件还涉及 `__CUDA_ARCH_FEAT_SM90_ALL`，不能把宏名称简单等同为所有 `sm_90` 编译目标均开启。

## 5. Example 63 的特殊价值

[示例源码](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/examples/63_hopper_gemm_with_weight_prefetch/63_hopper_gemm_with_weight_prefetch.cu)、[README](https://github.com/NVIDIA/cutlass/blob/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b/examples/63_hopper_gemm_with_weight_prefetch/README.md)。这是模型感知的权重预取示例：A 是可以提前读取的权重，将其预取到 L2；真正依赖前驱的激活仍需要等待。它使用示例自带的 collective builder 与 `kernel/sm90_gemm_tma_warpspecialized_with_prefetch.hpp`，不能把这种额外预取能力归给任意普通 GEMM。

源码第 62–65 行说明 overlap_ratio 越小越早触发，负值关闭 PDL；第 415、433 行将 `options.overlap_ratio >= 0` 作为 `launch_with_pdl`。prefetch_ratio 控制预取量；负值代表 best-effort，不代表关闭预取。README 还提示默认预取比例不一定合适。应分别讨论“普通 PDL 隐藏准备/启动开销”与“特殊 kernel 在等待区间预取独立权重”。

## 6. 可直接采用的审慎结论

PDL 放松的是 launch 的串行完成条件，而不是取消数据依赖。CUTLASS 把这种协议嵌入部分 kernel 的 producer / epilogue 路径，再由 host launch 开关授权调度器提前启动。它可以串成连续 kernel 链，但是否产生重叠、重叠多少以及是否提速，需要在具体硬件和问题规模上验证。本次为官方文档与固定版本源码核查，没有进行 GPU 性能测量。
