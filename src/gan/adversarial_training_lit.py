# src/gan/adversarial_training_lit.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.gan.losses import gradient_penalty

class AdversarialTrainingLit(pl.LightningModule):
    def __init__(self,
                 detector,
                 generator,
                 critic,
                 real_attack_data,
                 lr_det=0.0005,
                 lr_gen=0.0001,
                 lr_crit=0.0001,
                 n_critic=3,
                 lambda_gp=10,
                 anti_weight=0.1,
                 noise_dim=16,
                 cond_dim=1,
                 batch_size=512,
                 adv_ratio=0.5):
        super().__init__()
        self.detector = detector
        self.generator = generator
        self.critic = critic
        self.real_attack_data = real_attack_data

        self.lr_det = lr_det
        self.lr_gen = lr_gen
        self.lr_crit = lr_crit
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.anti_weight = anti_weight
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.batch_size = batch_size
        self.adv_ratio = adv_ratio

        # Detector loss: BCEWithLogitsLoss (combines sigmoid + BCE)
        self.criterion_det = nn.BCEWithLogitsLoss()
        # Anti-detection loss: BCELoss (expects probabilities)
        self.criterion_anti = nn.BCELoss()

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_det = torch.optim.Adam(self.detector.parameters(), lr=self.lr_det)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.9))
        opt_crit = torch.optim.Adam(self.critic.parameters(), lr=self.lr_crit, betas=(0.5, 0.9))
        return opt_det, opt_gen, opt_crit

    def training_step(self, batch, batch_idx):
        X_real, y_real = batch
        opt_det, opt_gen, opt_crit = self.optimizers()
        device = self.device

        # ─── 1. Train Critic ───
        for _ in range(self.n_critic):
            idx = torch.randint(0, self.real_attack_data.size(0), (self.batch_size,), device=device)
            real_att = self.real_attack_data[idx]
            cond_real = torch.ones(self.batch_size, 1, device=device)

            noise = torch.randn(self.batch_size, self.noise_dim, device=device)
            cond_fake = torch.ones(self.batch_size, 1, device=device)
            fake_att = self.generator(noise, cond_fake)

            c_real = self.critic(real_att, cond_real)
            c_fake = self.critic(fake_att.detach(), cond_fake)
            gp = gradient_penalty(self.critic, real_att, fake_att, cond_real, self.lambda_gp)
            c_loss = c_fake.mean() - c_real.mean() + gp

            opt_crit.zero_grad()
            self.manual_backward(c_loss)
            opt_crit.step()

        # ─── 2. Train Generator ───
        noise = torch.randn(self.batch_size, self.noise_dim, device=device)
        cond_fake = torch.ones(self.batch_size, 1, device=device)
        fake_att = self.generator(noise, cond_fake)

        # WGAN loss
        g_wgan = -self.critic(fake_att, cond_fake).mean()

        # Anti-detection loss: detector must output probability -> sigmoid
        det_logits = self.detector(fake_att)
        det_pred = torch.sigmoid(det_logits)
        target_benign = torch.zeros_like(det_pred)
        g_anti = self.criterion_anti(det_pred, target_benign)

        g_loss = g_wgan + self.anti_weight * g_anti

        opt_gen.zero_grad()
        self.manual_backward(g_loss)
        opt_gen.step()

        # ─── 3. Train Detector on mixed batch ───
        with torch.no_grad():
            noise = torch.randn(self.batch_size, self.noise_dim, device=device)
            cond_fake = torch.ones(self.batch_size, 1, device=device)
            fake_att_det = self.generator(noise, cond_fake)

        n_real = X_real.size(0)
        n_adv = int(self.batch_size * self.adv_ratio)
        if n_adv > 0:
            perm = torch.randperm(n_real, device=device)
            keep_idx = perm[:n_real - n_adv]
            X_combined = torch.cat([X_real[keep_idx], fake_att_det[:n_adv]], dim=0)
            y_combined = torch.cat([y_real[keep_idx], torch.ones(n_adv, 1, device=device)], dim=0)
        else:
            X_combined, y_combined = X_real, y_real

        # det_loss uses BCEWithLogitsLoss – takes raw logits
        det_logits = self.detector(X_combined)
        det_loss = self.criterion_det(det_logits, y_combined)

        opt_det.zero_grad()
        self.manual_backward(det_loss)
        opt_det.step()

        # Logging
        self.log("adv/critic_loss", c_loss, prog_bar=True)
        self.log("adv/gen_loss", g_loss, prog_bar=True)
        self.log("adv/det_loss", det_loss, prog_bar=True)
        self.log("adv/g_wgan", g_wgan, prog_bar=False)
        self.log("adv/g_anti", g_anti, prog_bar=False)

        # Accuracy (apply sigmoid to logits)
        with torch.no_grad():
            real_probs = torch.sigmoid(self.detector(X_real))
            acc_real = ((real_probs > 0.5) == y_real).float().mean()
            adv_probs = torch.sigmoid(self.detector(fake_att_det))
            acc_adv = ((adv_probs > 0.5) == torch.ones(self.batch_size, 1, device=device)).float().mean()
            self.log("adv/acc_real", acc_real, prog_bar=True)
            self.log("adv/acc_adv", acc_adv, prog_bar=True)