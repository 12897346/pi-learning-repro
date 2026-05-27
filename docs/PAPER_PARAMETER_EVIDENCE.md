# 参数与方法依据（防止“虚空赋值”）

本文档用于约束复现代码中的关键参数来源，避免无依据硬编码。

## 一、直接来自论文/补充材料

- `GAN normal 预训练 1000 epoch`、`physics-informed 微调 300 epoch`（正文叙述为判据满足时的停止点）  
  来源：正文 4. Experimental Section（Generator and Critic in GAN）。
- `GAN learning_rate = 1e-4`  
  来源：补充材料 Table S6。
- `Adam betas = (0.5, 0.9)`  
  来源：补充材料 Table S6 / S8 / S9。
- `critic_updates_per_g = 5`  
  来源：补充材料 Table S6。
- `lambda_gp = 10`  
  来源：补充材料 Eq.(14) 与 Table S6。
- `latent shape = 16 x 4 x 4 x 4`  
  来源：正文 Experimental Section + 补充材料 Table S6。
- `embedding size = 320`  
  来源：补充材料 Table S4 / S5 / S6。
- `generator/critic 通道结构`  
  来源：补充材料 Table S4 / S5。
- `phys-DNN 结构: input=1, hidden=50x2, output=7`  
  来源：补充材料 Table S9。
- `phys-CNN 输入 3+1`  
  来源：补充材料 Table S7（Input channel = 3+1）。
- `PSO 参数 c1=2, c2=2, w=0.8, particles=1000`  
  来源：正文 4.Experimental Section + 补充材料 Figure S5 说明。
- `GAN损失 L_G = L_G,org + gamma * MAE`  
  来源：正文 Experimental Section Eq.(1) 与补充材料 Eq.(16)。

## 二、实现中允许的“工程近似”（必须显式标注）

- **GAN 训练早停**：正文写明 normal / physics 在 **Wasserstein 距离 steady** 且**目测**形貌可接受时结束（1000 / 300 为常见上沿）。`paper_repro.gan_wdist_early_stop` 存 **window / std_tol / patience** 的常见工程默认（**非正文逐字超参**）；首轮跑通后请用 `metrics_*.csv`、`gan_exit_*.json` 再标定。`paper_repro.gan_physics_closs_std_tol` 为 physics 段 **c_loss** 同窗口稳态的启发式阈值（正文未给数，默认仅作试跑起点）。**目测**：`paper_repro.gan_preview_every` 或 CLI `--gan-preview-every` 写 `previews/*.png`（默认 0）；仍建议对照 `--save-every` 与最终权重。
- **PSO 早停**：正文写明「最优 J 连续 200 次迭代不变」为收敛判据。`forward_design_pso.py` 的 `--plateau-patience`（流水线默认 **200**，`0` 关闭）在 `gbest_J` 提升小于 `--plateau-tol` 时累计并提前结束；`forward_manifest.json` 记录 `stopped_by_plateau`。
- `strict TPB` 的 Python 逻辑实现为对论文逻辑算法（active cluster + 三相共边计数）的近似离散实现。  
- `fast/soft TPB` 仅用于训练加速或梯度近似，不作为论文“严格口径”结论依据。  
- 任何 fallback 数据构建参数只用于流程调试，不可用于论文数值结论。

## 三、执行约束

- 训练脚本默认走 `strict`（论文口径）并输出指标文件，避免只看图片不看收敛证据。
- 图中分组标签必须来自实际样本统计，不允许写死范围冒充论文原始范围。
- 若使用 OpenFOAM 场图（Figure 7 d/e/f），必须来自真实仿真导出，不可用随机占位数据。

## 四、各模块「何时停止训练/优化」与原文一一对照

下表「原文」指 Adv. Energy Mater. 2023, 2300244 **正文** 4. Experimental Section（及已核对的补充材料中与本流程直接相关的表/式）。**符合性**指「判据是否覆盖原文要点」而非数值逐比特一致。

| 模块 | 脚本 / 入口 | 原文停训/收敛依据（摘要） | 本仓库实现 | 符合性 |
|------|----------------|-----------------------------|-------------|--------|
| **phys-DNN** | `train_phys_models.py` | 正文：L1、Adam 等；**约 100 epoch 内**达到稳定收敛且 MAE&lt;1（叙述性结果）。 | `train_epochs` 上沿 + **`--paper-early-stop`**：`phys_dnn_val_mae_early_stop`（正文 MAE 叙述）+ `phys_surrogate_plateau_*`（**常见平台启发式**，首轮默认见 YAML，跑完 `metrics.json` 再调）。`--phys-disable-early-stop` 可关。 | **基本符合**：文献项与工程项在 YAML 中分层注释。 |
| **phys-CNN**（及文中 purely data-driven CNN） | `train_phys_models.py` | 正文：**约 70 epoch** 内对 MAE 达到足够精度（steady convergence）。 | `train_epochs` 上沿 + 同上；**不设** CNN 的 MAE&lt;1；与 DNN 共用平台键（启发式）。 | **基本符合**：同上。 |
| **GAN normal** | `train_gan_fallback.py` `--training-stage normal` | 正文：**Wasserstein 距离变 steady** 且**目测**形貌可接受时停；并叙述常在 **1000 epoch** 量级结束。 | **`--epochs` 上沿** + **`paper_repro.gan_wdist_early_stop`**（由流水线注入 CLI 或本地默认读 YAML）：w_dist 批均值滑动 std + patience；可选 **`--gan-preview-every`**。`gan_exit_normal.json`。 | **部分**：判据数值集中在 YAML 并标注为工程代理；**目测**仍靠人工。 |
| **GAN physics** | `train_gan_fallback.py` `--training-stage physics` | 正文：**Wasserstein 与目测均满足**时停；叙述约 **300 epoch**。 | w_dist 判据同 YAML；**c_loss** 默认读 `gan_physics_closs_std_tol`（**试跑启发式**，可改 YAML）；`--physics-skip-closs-stable` / `--gan-physics-skip-closs-stable` 可关第二项。 | **部分**：第二项为工程代理，标定依赖训练日志。 |
| **PSO（forward design）** | `forward_design_pso.py` | 正文：**一般 200–300 次迭代**内稳定；**最优 J 连续 200 次迭代不变**则视为收敛；粒子 **1000**；先验 **10 000** 样本。 | **`--iters` 上沿**（默认 300）+ **`--plateau-patience`**（流水线默认 **200**，`gbest` 提升 &lt; `plateau-tol` 计平台步）；先验 `prior_n` 与文一致量级；`forward_manifest.json` 记 `stopped_by_plateau`。 | **基本符合**：迭代上沿与「200 步不变」已编码；与原文差在 **浮点容差** 及未显式区分「200–300」内的自适应上限。 |
| **phys-CNN 混合重训**（可选反馈链） | `train_phys_models.py --model phys_cnn` | 正文对「混合后再训 CNN」的停训判据**未单独开条**；与主 phys-CNN 同一类监督学习叙述。 | 与主 CNN 相同：`--epochs` / yaml 上沿 + 流水线默认的 `--paper-early-stop`（仅平台项作用于 CNN）。 | **同 phys-CNN**：与主步一致的动态判据。 |

### 结论（用于自检）

- **已较贴近原文、且可自动化**的：**PSO**（gbest 平台 200）；**GAN**（`gan_wdist_early_stop` + 可选 `gan_physics_closs_std_tol`，均为 YAML 声明的试跑默认，**标定靠日志**）；**phys surrogate**（`--paper-early-stop`：MAE 阈值 + 平台键见 YAML）。  
- **与原文仍有差距、须人工的**：**GAN 目测**（`previews/`、`--save-every`、最终权重）。  
- **文献核对**：本仓库无法在自动化环境中稳定拉取 Wiley 全文/SI（常 403）；请以本地 PDF/SI 或机构订阅为准，修改 `configs/paper_params.yaml` 中各键并保留你的核对备注。  
- 调参与关闭：`GAN_WDIST_STABLE_PATIENCE=0`、`PSO_PLATEAU_PATIENCE=0`、`PHYS_DISABLE_EARLY_STOP=1`、`GAN_PHYSICS_SKIP_CLOSS=1` 等见 `scripts/slurm_run_paper_pipeline.sh` 注释。

