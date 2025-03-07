
What is the Neural Processing Unit?
The Neural Processing Unit (NPU) is an FPGA soft processor (i.e., overlay) architecture for low latency, low batch AI inference. It adopts the "persistent AI" approach, in which all model weights are kept persistent in the on-chip SRAM memory of one or more network-connected FPGAs to eliminate the expensive off-chip memory accesses. The NPU is a domain-specific software-programmable processor. Therefore, once the NPU bitstream is compiled and deployed on an FPGA, users can rapidly program it to run different AI workloads using a high-level domain-specific language or a deep learning framework (e.g. TensorFlow Keras) purely in software. This approach enables AI application developers to use FPGAs for AI inference acceleration without the need for FPGA design expertise or suffering from the long runtime of FPGA CAD tools.

The current version of the NPU supports a variety of AI inference workloads such as multi-layer perceptron (MLP), recurrent neural network (RNN), gated recurrent unit (GRU), and long short-term memory (LSTM) models. All these models are more memory-bound with low data reuse, which benefit greatly from the tremendous on-chip bandwidth between the SRAM memories and compute units in the FPGA. However, the NPU architecture and ISA are extendable to support other AI workloads.

The NPU Framework
The NPU framework consists of:

NPU Overlay: This is the hardware implementation of the NPU architecture coded in SystemVerilog RTL and optimized for Intel's Stratix 10 NX FPGA. This architecture is highly optimized to generate a single high-quality bitstream, which is deployed on the FPGA and can be programmed with different workloads purely from software.
Instruction Set Architecture (ISA): This is the intermediate layer between the NPU hardware and its software stack. The NPU uses a very long instruction word (VLIW), which we refer to as an "instruction chain", that controls a number of coarse-grained chained processing units. Each instruction chain can trigger the execution of thousands of operations, similar in nature to CISC instructions.
NPU Compiler: This translates a user input workload written as a Tensorflow Keras sequential model into an NPU executable binary. The NPU compiler implements low-level APIs for each of the operations supported by the NPU architecture (e.g., matrix-vector multiplication, addition, activation, etc.). Then, it builds Keras APIs for supported models/layers (e.g., Dense, RNN, GRU, LSTM) using these low-level APIs. The output of the compiler is binary VLIW instructions that can be sent and executed on the NPU overlay.
Functional Simulator: This simulates the compiled NPU instructions functionally to verify that they implement the intended functionality and generate golden results for C++ and RTL simulation.
C++ Simulator: This is a detailed NPU performance simulator written in C++ to provide fast and reliable performance estimates for running the compiled NPU program on an NPU architecture with specific architecture parameters. This can be used for rapid exploration of the NPU overlay design space and NPU program optimization.
RTL Simulation: This is the slowest but most accurate simulation flow to obtain performance results of a compiled NPU program when executed on a given NPU instance.
Repository Description
The repository consists of the following directories:

compiler: includes the NPU front-end (API and compiler) that users can use to write NPU workloads as Tensorflow Keras Sequential models
rtl: includes the RTL implementation of the NPU hardware for the Stratix 10 NX FPGA
scripts: includes testing scripts for C++ and RTL simulation of the NPU benchmark suite used in the FPT'20 paper
simulator: includes the NPU C++ simulator used for fast NPU performance estimation and architecture exploration
List of NPU-related Publications
A. Boutros, E. Nurvitadhi, and V. Betz. "Specializing for Efficiency: Customizing AI Inference Processors on FPGAs". In the IEEE International Conference on Microelectronics (ICM), 2021.
A. Boutros, E. Nurvitadhi, R. Ma, S. Gribok, Z. Zhao, J. Hoe, V. Betz, and M. Langhammer. "Beyond Peak Performance: Comparing the Real Performance of AI-Optimized FPGAs and GPUs". In the IEEE International Conference on Field-Programmable Technology (FPT), 2020.
D. Kwon, S. Hur, H. Jang, E. Nurvitadhi, J. Kim. "Scalable Multi-FPGA Acceleration for Large RNNs with Full Parallelism Levels". In the ACM/IEEE Design Automation Conference (DAC), 2020.
E. Nurvitadhi, A. Boutros, P. Budhkar, A. Jafari, D. Kwon, D. Sheffield, A. Prabhakaran, K. Gururaj, P. Appana, and M. Naik. "Scalable Low-Latency Persistent Neural Machine Translation on CPU Server with Multiple FPGAs". In the IEEE International Conference on Field-Programmable Technology (FPT), 2019.
E. Nurvitadhi, D. Kwon, A. Jafari, A. Boutros, J. Sim, P. Tomson, H. Sumbul, G. Chen, P. Knag, R. Kumar, R. Krishnamurthy, S. Gribok, B. Pasca, M. Langhammer, D. Marr, and A. Dasu. "Why Compete When You Can Work Together: FPGA-ASIC Integration for Persistent RNNs". In the IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2019.
E. Nurvitadhi, D. Kwon, A. Jafari, A. Boutros, J. Sim, P. Tomson, H. Sumbul, G. Chen, P. Knag, R. Kumar, R. Krishnamurthy, S. Gribok, B. Pasca, M. Langhammer, D. Marr, and A. Dasu. "Evaluating and Enhancing Intel Stratix 10 FPGAs for Persistent Real-Time AI". In the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA), 2019.
Citation
If you use the NPU code in this repo for your research, please cite the following paper:

A. Boutros, E. Nurvitadhi, R. Ma, S. Gribok, Z. Zhao, J. Hoe, V. Betz, and M. Langhammer. "Beyond Peak Performance: Comparing the Real Performance of AI-Optimized FPGAs and GPUs". In the IEEE International Conference on Field-Programmable Technology (FPT), 2020.
You can use the following BibTex entry:

@article{npu_s10_nx,
  title={{Beyond Peak Performance: Comparing the Real Performance of AI-Optimized FPGAs and GPUs}},
  author={Boutros, Andrew and others},
  booktitle={IEEE International Conference on Field-Programmable Technology (ICFPT)},
  year={2020}
}
Pages 1
Find a page…
Home
What is the Neural Processing Unit?
The NPU Framework
Repository Description
List of NPU-related Publications
Citation
Clone this wiki locally
https://github.com/intel/fpga-npu.wiki.git
Footer
© 2025 GitHub, In