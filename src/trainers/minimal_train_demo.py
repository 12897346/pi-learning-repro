from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml

from src.losses.pi_loss import compute_gradient_penalty, critic_wgan_gp_loss, generator_loss_with_physics
from src.models.critic2d import Critic2D
from src.models.generator3d import Generator3D
from src.models.phys_dnn import PhysDNN
from src.utils.seed import set_seed


@dataclass
class DemoConfig:
    device: str = "cpu"
    steps: int = 2
    batch_size_override: int | None = 1


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_minimal_demo(config_path: str, demo_cfg: DemoConfig) -> None:
    set_seed(42)
    p = load_params(config_path)
    device = torch.device(demo_cfg.device)

    gan_cfg = p["gan"]
    batch_size = demo_cfg.batch_size_override or gan_cfg["batch_size"]
    zc = gan_cfg["latent_channels"]
    classes = gan_cfg["classes_inverse"]
    use_tiny = bool(gan_cfg.get("debug_tiny", False))
    g_channels = gan_cfg["tiny_channels"]["generator"] if use_tiny else gan_cfg["paper_channels"]["generator"]
    c_channels = gan_cfg["tiny_channels"]["critic"] if use_tiny else gan_cfg["paper_channels"]["critic"]

    gen = Generator3D(
        z_channels=zc,
        num_classes=classes,
        embed_dim=gan_cfg["embedding_size"],
        channels=g_channels,
    ).to(device)
    critic = Critic2D(
        in_channels=3,
        num_classes=classes,
        embed_dim=gan_cfg["embedding_size"],
        channels=c_channels,
    ).to(device)
    phys = PhysDNN(
        input_size=p["phys_dnn"]["input_size"],
        hidden_dim=p["phys_dnn"]["hidden_dim"],
        output_size=p["phys_dnn"]["output_size"],
    ).to(device)
    # demo 中把 phys 固定为预训练替代器（这里只是示意）
    phys.eval()
    for param in phys.parameters():
        param.requires_grad_(False)

    opt_g = torch.optim.Adam(gen.parameters(), lr=gan_cfg["learning_rate"], betas=tuple(gan_cfg["adam_betas"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=gan_cfg["learning_rate"], betas=tuple(gan_cfg["adam_betas"]))
    l1 = nn.L1Loss()

    print("=== 最小复现 Demo 开始（轻量模式） ===")
    print(f"device={device}, steps={demo_cfg.steps}, batch_size={batch_size}, debug_tiny={use_tiny}")
    for step in range(demo_cfg.steps):
        labels = torch.randint(0, classes, (batch_size,), device=device)
        z = torch.randn(batch_size, zc, 4, 4, 4, device=device)
        real = torch.rand(batch_size, 3, 64, 64, 64, device=device)

        # 1) 更新 critic
        fake = gen(z, labels).detach()
        real_score = critic(real, labels)
        fake_score = critic(fake, labels)
        gp = compute_gradient_penalty(critic, real, fake, labels, device)
        c_loss = critic_wgan_gp_loss(real_score, fake_score, gp, lambda_gp=gan_cfg["lambda_gp"])

        opt_c.zero_grad()
        c_loss.backward()
        opt_c.step()

        # 2) 更新 generator（含 physics loss）
        fake = gen(z, labels)
        fake_score = critic(fake, labels)

        # 用简化 TPB 标量模拟 phys-DNN 输入，保证流程可跑通
        tpb_scalar = fake[:, 1:2].mean(dim=(2, 3, 4)).reshape(batch_size, 1)
        pred_j = phys(tpb_scalar)
        target_j = torch.rand_like(pred_j)

        g_total, g_org, g_phys = generator_loss_with_physics(
            fake_score, pred_j, target_j, gamma_phys=gan_cfg["gamma_phys"]
        )
        # 增加一个极小监督项，避免 demo 全靠随机张量导致无意义梯度震荡
        g_total = g_total + 0.01 * l1(pred_j, target_j)

        opt_g.zero_grad()
        g_total.backward()
        opt_g.step()

        print(
            f"step={step + 1} "
            f"c_loss={c_loss.item():.4f} "
            f"g_total={g_total.item():.4f} "
            f"g_org={g_org.item():.4f} "
            f"g_phys={g_phys.item():.4f}"
        )

    print("=== 最小复现 Demo 完成 ===")
