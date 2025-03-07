# Intel开源NPU设计

# 为什么选择这个NPU
软件 + compiler + 硬件配置比较全，适合作为一个全流程设计参考

# 文档
[链接](./doc/wiki.md)

# 运行方式
```
bash run.sh
```


# 已知存在的问题

1. performance model running simulation ... 时CPU几乎没有占用，等了非常久
1. 在compiler.py中官方提到了NPU前端没有MFU(数学函数单元)中非线性激活函数的软件实现等一系列架构的问题

