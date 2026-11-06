# NVIDIA PDL dissect

- [完整中文分析、实验过程与数据](docs/pdl-analysis.md)
- [CUDA/PTX 官方语义与 CUTLASS v4.3.5 固定源码证据](docs/pdl-official-research.md)
- [原始 demo（保留原样，包含已记录的正确性问题）](demo.cu)
- [正确性/PDL/预取/三 kernel 链/多 stream 实验](experiments/pdl_probe.cu)
- [全配置实测汇总](results/summary.csv)
- [逐次 CSV 统计脚本](experiments/summarize.py)

实验平台：B300 CC 10.3 / CUDA 13.0。逐次原始数据与 Nsight 文件保存在报告列出的远程路径；本仓库中的 summary 是汇总，不冒充原始测量。
