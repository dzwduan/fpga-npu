from matplotlib import pyplot as plt

# 读取基准文件
baseline_file = "/home/dzw/fpga-npu/scripts/perf_baseline"
workloads = []
results = []

with open(baseline_file, "r") as file:
    for line in file:
        split_line = line.split()
        workloads.append(split_line[0])
        results.append(float(split_line[1]))

# 绘制折线图
plt.figure(figsize=(10, 5))
plt.plot(workloads, results, marker="o", linestyle="-", color="b")
plt.xlabel("Workload")
plt.ylabel("Performance (TOPS)")
plt.title("Performance Baseline")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
