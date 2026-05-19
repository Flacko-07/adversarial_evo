# src/gan/lit_module.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.gan.losses import gradient_penalty

class GANLightning(pl.LightningModule):
    def __init__(self, generator, critic, detector_frozen,
                 lr_gen, lr_crit, n_critic, lambda_gp,
                 anti_weight, noise_dim, cond_dim, batch_size):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.detector = detector_frozen
        self.lr_gen = lr_gen
        self.lr_crit = lr_crit
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.anti_weight = anti_weight
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.batch_size = batch_size

        # Binary cross‑entropy for anti‑detection loss
        self.criterion_anti = nn.BCELoss()

        self.automatic_optimization = False
        self.real_attack_data = None

    def set_real_attack_data(self, real_data):
        self.real_attack_data = real_data

    def training_step(self, batch, batch_idx):
        if self.real_attack_data is None:
            raise ValueError("Must call set_real_attack_data() before training.")

        opt_gen, opt_crit = self.optimizers()
        device = self.device

        # ---- Train Critic ----
        for _ in range(self.n_critic):
            idx = torch.randint(0, self.real_attack_data.size(0), (self.batch_size,), device=device)
            real_samples = self.real_attack_data[idx]
            cond_real = torch.ones(self.batch_size, 1, device=device)

            noise = torch.randn(self.batch_size, self.noise_dim, device=device)
            cond_fake = torch.ones(self.batch_size, 1, device=device)
            fake_samples = self.generator(noise, cond_fake)

            c_real = self.critic(real_samples, cond_real)
            c_fake = self.critic(fake_samples.detach(), cond_fake)

            c_loss = c_fake.mean() - c_real.mean()
            gp = gradient_penalty(self.critic, real_samples, fake_samples, cond_real, self.lambda_gp)
            total_c_loss = c_loss + gp

            opt_crit.zero_grad()
            self.manual_backward(total_c_loss)
            opt_crit.step()

        # ---- Train Generator ----
        noise = torch.randn(self.batch_size, self.noise_dim, device=device)
        cond_fake = torch.ones(self.batch_size, 1, device=device)
        fake_samples = self.generator(noise, cond_fake)

        g_wgan = -self.critic(fake_samples, cond_fake).mean()

        # Anti‑detection loss: fool the detector into predicting "benign"
        det_pred_logits = self.detector(fake_samples)
        det_pred = torch.sigmoid(det_pred_logits)
        target_benign = torch.zeros_like(det_pred)
        g_anti = self.criterion_anti(det_pred, target_benign)

        g_loss = g_wgan + self.anti_weight * g_anti

        opt_gen.zero_grad()
        self.manual_backward(g_loss)
        opt_gen.step()

        self.log("gan/crit_loss", total_c_loss, prog_bar=True)
        self.log("gan/gen_loss", g_loss, prog_bar=True)
        self.log("gan/wgan", g_wgan, prog_bar=False)
        self.log("gan/anti", g_anti, prog_bar=False)

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.9))
        opt_crit = torch.optim.Adam(self.critic.parameters(), lr=self.lr_crit, betas=(0.5, 0.9))
        return opt_gen, opt_crit