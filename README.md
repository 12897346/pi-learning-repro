# π Learning 复现（分步执行）

## 当前状态

这是论文复现的第 1 步：先搭建可执行框架，不盲目开训。

> 关键事实：论文核心数据需向作者申请，当前无法做“严格同结果复现”。

## 复现路线图

1. 第 1 步（已开始）：环境与任务基线搭建
2. 第 2 步：数据规范与预处理（3 相体素、连通性、TPB）
3. 第 3 步：训练 phys-DNN / phys-CNN（电流密度 surrogate）
4. 第 4 步：训练 cWGAN-GP 与 performance-informed 微调
5. 第 5 步：inverse design（目标 J 区间生成）
6. 第 6 步：forward design（PSO 搜索最优潜空间）
7. 第 7 步：评估与对照（J 分布、误差、结构统计）

## 第 1 步你现在要做什么

### A. 准备 Python 环境

建议 Python 3.10/3.11，安装依赖：

```powershell
pip install -r requirements.txt
```

### B. 执行环境检查

```powershell
python scripts/check_env.py
```

### C. 验收标准

- 能打印 Python、PyTorch、NumPy、CUDA 信息
- 不报错退出（退出码 0）


