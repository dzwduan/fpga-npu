# 官方Wiki中文版

## 什么是神经处理单元（NPU）
神经处理单元（NPU）是一种基于FPGA的软核处理器（即覆盖层架构），专为低延迟、小批量AI推理设计。其采用"持久化AI"方法，将所有权重参数持久化存储在一个或多个网络互联FPGA的片上SRAM存储器中，以消除昂贵的片外存储器访问。NPU属于领域专用型软件可编程处理器。因此，一旦NPU比特流在FPGA上完成编译部署，用户即可通过高级领域专用语言或深度学习框架（如TensorFlow Keras）在纯软件层面快速编程运行不同AI工作负载。该方法使AI应用开发者无需FPGA设计专业知识或承受FPGA CAD工具的长时延，即可利用FPGA进行AI推理加速。

当前版本NPU支持多种AI推理负载，包括多层感知机（MLP）、循环神经网络（RNN）、门控循环单元（GRU）和长短期记忆（LSTM）模型。这些模型普遍具有高存储带宽需求与低数据复用特性，因此能够充分利用FPGA中SRAM存储器与计算单元之间的片上超大带宽优势。NPU架构及其指令集（ISA）具有可扩展性，可支持更多AI工作负载。

## NPU框架体系
NPU框架包含以下核心组件：
1. NPU覆盖层：基于SystemVerilog RTL实现的NPU硬件架构，针对Intel Stratix 10 NX FPGA进行深度优化。该架构经高度优化可生成单一高质量比特流，部署于FPGA后支持纯软件编程实现不同工作负载。

1. 指令集架构（ISA）：作为NPU硬件与软件栈之间的中间层。NPU采用超长指令字（VLIW）架构，我们称之为"指令链"，用于控制多个粗粒度级联处理单元。每条指令链可触发数千次运算操作，其本质类似于CISC指令。

1. NPU编译器：将用户以TensorFlow Keras顺序模型编写的工作负载转换为NPU可执行二进制文件。该编译器为NPU架构支持的每个运算（如矩阵向量乘法、加法、激活等）实现底层API，并基于这些API构建支持模型/层（如Dense、RNN、GRU、LSTM）的Keras接口。最终输出可被NPU覆盖层执行的VLIW二进制指令。

1. 功能模拟器：对编译后的NPU指令进行功能级仿真，验证其功能正确性并生成用于C++和RTL仿真的黄金参考结果。

1. C++模拟器：采用C++编写的细粒度NPU性能模拟器，可快速评估编译后NPU程序在特定架构参数下的运行性能。该工具支持NPU覆盖层设计空间探索和程序优化。

1. RTL仿真：最精确但耗时最长的仿真流程，用于获取编译后NPU程序在指定NPU实例上的精确性能数据。

## 代码仓库结构
本仓库包含以下目录：
1. compiler：包含NPU前端（API和编译器），用户可通过TensorFlow Keras顺序模型编写NPU工作负载
1. rtl：包含针对Stratix 10 NX FPGA的NPU硬件RTL实现
1. scripts：包含用于FPT'20论文中NPU基准测试套件的C++和RTL仿真脚本
1. simulator：包含用于快速性能评估和架构探索的NPU C++模拟器
1. patch: 用于运行真实FPGA的，需要设备

## 架构相关

1. [A_Configurable_Cloud-Scale_DNN_Processor_for_Real-Time_AI](../doc/A_Configurable_Cloud-Scale_DNN_Processor_for_Real-Time_AI.pdf)
1. [Serving DNNs in RealTime at Datacenter Scale with Project Brainwave](../doc/mi0218_Chung-2018Mar25.pdf)
1. [Brainwave-Datacenter-Chung-Microsoft](../doc/HC29.22622-Brainwave-Datacenter-Chung-Microsoft-2017_08_11_2017.compressed.pdf)
1. [support GNN](../doc/A_Software-Programmable_Neural_Processing_Unit_for_Graph_Neural_Network_Inference_on_FPGAs.pdf)
1. [Hotchips31](../doc/HC31_T2_Microsoft_CarrieChiouChung.pdf)
1. [老石谈芯article](https://shilicon.com/archives/180)
1. [cs217](https://zhuanlan.zhihu.com/p/329789414)


## List of NPU-related Publications
1. [Boutros, E. Nurvitadhi, and V. Betz. "Specializing for Efficiency: Customizing AI Inference Processors on FPGAs". In the IEEE International Conference on Microelectronics (ICM), 2021](../doc/01_icm2021_specialization.pdf)

1. [Boutros, E. Nurvitadhi, R. Ma, S. Gribok, Z. Zhao, J. Hoe, V. Betz, and M. Langhammer. "Beyond Peak Performance: Comparing the Real Performance of AI-Optimized FPGAs and GPUs". In the IEEE International Conference on Field-Programmable Technology (FPT), 2020.](../doc/02_beyond-peak-performance-white-paper.pdf)

1. [Kwon, S. Hur, H. Jang, E. Nurvitadhi, J. Kim. "Scalable Multi-FPGA Acceleration for Large RNNs with Full Parallelism Levels". In the ACM/IEEE Design Automation Conference (DAC), 2020.](../doc/03_Scalable_Multi-FPGA_Acceleration_for_Large_RNNs_with_Full_Parallelism_Levels.pdf)

1. [Nurvitadhi, A. Boutros, P. Budhkar, A. Jafari, D. Kwon, D. Sheffield, A. Prabhakaran, K. Gururaj, P. Appana, and M. Naik. "Scalable Low-Latency Persistent Neural Machine Translation on CPU Server with Multiple FPGAs". In the IEEE International Conference on Field-Programmable Technology (FPT), 2019.](../doc/04_Scalable_Low-Latency_Persistent_Neural_Machine_Translation_on_CPU_Server_with_Multiple_FPGAs.pdf)

1. [Nurvitadhi, D. Kwon, A. Jafari, A. Boutros, J. Sim, P. Tomson, H. Sumbul, G. Chen, P. Knag, R. Kumar, R. Krishnamurthy, S. Gribok, B. Pasca, M. Langhammer, D. Marr, and A. Dasu. "Why Compete When You Can Work Together: FPGA-ASIC Integration for Persistent RNNs". In the IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2019.](../doc/05_Why_Compete_When_You_Can_Work_Together_FPGA-ASIC_Integration_for_Persistent_RNNs.pdf)

1. [Nurvitadhi, D. Kwon, A. Jafari, A. Boutros, J. Sim, P. Tomson, H. Sumbul, G. Chen, P. Knag, R. Kumar, R. Krishnamurthy, S. Gribok, B. Pasca, M. Langhammer, D. Marr, and A. Dasu. "Evaluating and Enhancing Intel Stratix 10 FPGAs for Persistent Real-Time AI". In the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA), 2019.](../doc/06_Evaluating_The_Highly-Pipelined_Intel_Stratix_10_FPGA_Architecture_Using_Open-Source_Benchmarks.pdf)

1. [nside Project Brainwave’s Cloud-Scale, Real-Time AI Processor](../doc/07_Inside_Project_Brainwaves_Cloud-Scale_Real-Time_AI_Processor.pdf)

## Citation
If you use the NPU code in this repo for your research, please cite the following paper:

A. Boutros, E. Nurvitadhi, R. Ma, S. Gribok, Z. Zhao, J. Hoe, V. Betz, and M. Langhammer. "Beyond Peak Performance: Comparing the Real Performance of AI-Optimized FPGAs and GPUs". In the IEEE International Conference on Field-Programmable Technology (FPT), 2020.
You can use the following BibTex entry:
