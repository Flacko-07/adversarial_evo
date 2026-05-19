import torch

def gradient_penalty(critic, real_samples, fake_samples, condition, lambda_gp=10):
    """WGAN-GP gradient penalty."""
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, device=real_samples.device).expand_as(real_samples)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)

    d_interpolated = critic(interpolated, condition)
    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return penalty
