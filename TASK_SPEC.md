# π Learning 论文复现任务规格（S1/S2）

## 1. 目标范围

- 目标论文：`π Learning: A Performance-Informed Framework for Microstructural Electrode Design`
- 复现目标（分阶段）：
  - P1：可运行训练与评估框架（环境、目录、脚本、日志规范）
  - P2：复现 surrogate（phys-CNN / phys-DNN）训练流程
  - P3：复现 performance-informed cWGAN-GP 训练流程
  - P4：复现 inverse design 与 forward design（PSO）流程

## 2. 明确约束

- 论文声明核心数据需向作者申请；当前仓库无原始 PFIB-SEM 数据。
- 因此当前只能进行“流程复现 + 近似结果复现”，不能承诺数值逐点一致。
- 若要严格复现，必须补齐：
  - 原始/同源 3D 多相微结构数据；
  - 补充材料中的网络结构超参数表（S4-S9）；
  - 多物理场 OpenFOAM 标注流程或等价标签数据。

## 3. 验收标准（阶段一）

- 可在本机执行环境检查脚本并通过。
- 目录结构满足后续模块化开发。
- 形成可执行实验计划（包含每步输入、输出、评估指标）。

## 4. 非目标

- 本阶段不直接训练大模型。
- 本阶段不生成“论文同数值”图表结论。

## 5. 风险与假设

- 风险：数据不可得导致结果不可比。
- 假设：可先用你已有 3D 数据完成“方法链路复现”，后续再替换为目标数据集。
