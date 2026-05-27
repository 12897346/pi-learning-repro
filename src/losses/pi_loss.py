import torch


def generator_loss_with_physics(
    critic_fake_score: torch.Tensor,
    predicted_j: torch.Tensor,
    target_j: torch.Tensor,
    gamma_phys: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    论文 Eq.16 近似实现:
    L_G = L_G_org + gamma * MAE(predicted_j, target_j)
    """
    l_g_org = -critic_fake_score.mean()
    l_phys = torch.mean(torch.abs(predicted_j - target_j))
    l_total = l_g_org + gamma_phys * l_phys
    return l_total, l_g_org, l_phys


def j_curve_continuity_loss(predicted_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    J-η 曲线连续性约束（离散二阶差分）+ 单调性约束（随 η 不下降）。
    predicted_j: [B, 7]
    """
    if predicted_j.ndim != 2 or predicted_j.shape[1] < 3:
        raise ValueError(f"predicted_j 期望 [B,>=3]，当前 {predicted_j.shape}")
    first_diff = predicted_j[:, 1:] - predicted_j[:, :-1]
    second_diff = first_diff[:, 1:] - first_diff[:, :-1]
    l_smooth = torch.mean(torch.abs(second_diff))
    # 非单调（下降）部分惩罚
    l_mono = torch.mean(torch.relu(-first_diff))
    return l_smooth, l_mono


def critic_wgan_gp_loss(
    critic_real_score: torch.Tensor,
    critic_fake_score: torch.Tensor,
    gradient_penalty: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    # WGAN-GP: E[fake] - E[real] + lambda * GP
    return critic_fake_score.mean() - critic_real_score.mean() + lambda_gp * gradient_penalty


def compute_gradient_penalty(
    critic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates, labels)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
