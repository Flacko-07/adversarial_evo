# src/evolution/loop.py
import pandas as pd  
import copy, torch, pytorch_lightning as pl
from src.data import load_data, get_dataloaders, real_attack_tensor
from src.detector.lit_module import DetectorLit
from src.gan.lit_module import GANLightning
from hydra.utils import instantiate

class AdversarialLoop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg.seed
        pl.seed_everything(self.seed)
        self.train_loader, self.test_loader, self.real_attack, self.input_dim = load_data(cfg, seed=self.seed)
        self.history_test_acc = []
        self.history_adv_acc = []
        self.detector = None
        self.gan_module = None
        self.generator = None

    def train_initial_detector(self):
        model = instantiate(self.cfg.detector.model, input_dim=self.input_dim)
        self.detector = DetectorLit(model, lr=self.cfg.detector.lr)
        trainer = pl.Trainer(**self.cfg.trainer, logger=False, enable_checkpointing=False)
        trainer.fit(self.detector, self.train_loader, self.test_loader)
        return self.detector

    def freeze_detector_for_gan(self):
        frozen = copy.deepcopy(self.detector.model)
        for param in frozen.parameters():
            param.requires_grad = False
        frozen.eval()
        return frozen

    def train_gan(self, frozen_detector, anti_weight=None):
        if anti_weight is None:
            anti_weight = self.cfg.gan.anti_weight
        generator = instantiate(self.cfg.gan.generator, output_dim=self.input_dim)
        critic = instantiate(self.cfg.gan.critic, input_dim=self.input_dim)
        self.gan_module = GANLightning(
            generator=generator, critic=critic, detector_frozen=frozen_detector,
            lr_gen=self.cfg.gan.lr_gen, lr_crit=self.cfg.gan.lr_crit,
            n_critic=self.cfg.gan.n_critic, lambda_gp=self.cfg.gan.lambda_gp,
            anti_weight=anti_weight, noise_dim=self.cfg.gan.noise_dim,
            cond_dim=self.cfg.gan.cond_dim, batch_size=self.cfg.data.batch_size
        )
        self.gan_module.set_real_attack_data(self.real_attack.to(self.gan_module.device))
        dummy_ds = torch.utils.data.TensorDataset(torch.zeros(1,1))
        dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=1)
        trainer = pl.Trainer(
            max_epochs=self.cfg.gan.epochs,
            accelerator=self.cfg.trainer.accelerator, devices=self.cfg.trainer.devices,
            precision=self.cfg.trainer.precision, logger=False, enable_checkpointing=False,
            enable_progress_bar=self.cfg.trainer.enable_progress_bar,
            log_every_n_steps=self.cfg.trainer.log_every_n_steps, limit_train_batches=1
        )
        trainer.fit(self.gan_module, train_dataloaders=dummy_loader)
        self.generator = generator
        return generator

    def generate_adversarial_samples(self, n):
        self.generator.eval()
        device = next(self.generator.parameters()).device
        with torch.no_grad():
            noise = torch.randn(n, self.cfg.gan.noise_dim, device=device)
            cond = torch.ones(n, 1, device=device)
            samples = self.generator(noise, cond)
        return samples.cpu()

    def test_accuracy(self):
        device = next(self.detector.model.parameters()).device
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self.detector.model(xb) > 0.0
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

    def adv_accuracy(self, adv_samples):
        device = next(self.detector.model.parameters()).device
        adv_tensor = adv_samples.to(device)
        labels = torch.ones(adv_tensor.size(0), 1, device=device)
        with torch.no_grad():
            preds = self.detector.model(adv_tensor) > 0.5
            acc = (preds == labels).float().mean().item()
        return acc

    def retrain_detector(self, adv_samples, additional_epochs):
        train_ds = self.train_loader.dataset
        X_orig, y_orig = train_ds.tensors[0], train_ds.tensors[1]
        X_aug = torch.cat([X_orig, adv_samples], dim=0)
        y_aug = torch.cat([y_orig, torch.ones(adv_samples.size(0), 1)], dim=0)
        aug_loader = get_dataloaders(
            X_aug.numpy(), y_aug.numpy().flatten(),
            X_aug.numpy()[:100], y_aug.numpy()[:100],
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.get("pin_memory", False)
        )[0]
        model = instantiate(self.cfg.detector.model, input_dim=self.input_dim)
        self.detector = DetectorLit(model, lr=self.cfg.detector.lr)
        trainer = pl.Trainer(
            max_epochs=additional_epochs,
            **{k:v for k,v in self.cfg.trainer.items() if k!='max_epochs'},
            logger=False, enable_checkpointing=False
        )
        trainer.fit(self.detector, aug_loader, self.test_loader)
        return self.detector.model

    def run(self):
        print("=== Phase 1: Train initial Blue detector ===")
        self.train_initial_detector()
        print("=== Phase 2: Train initial GAN against Blue ===")
        frozen = self.freeze_detector_for_gan()
        self.train_gan(frozen, anti_weight=self.cfg.gan.anti_weight)

        for cycle in range(self.cfg.loop.n_cycles):
            print(f"\n=== Cycle {cycle+1}/{self.cfg.loop.n_cycles} ===")
            test_acc = self.test_accuracy()
            adv_samples = self.generate_adversarial_samples(self.cfg.loop.adv_samples_per_cycle)
            adv_acc = self.adv_accuracy(adv_samples)
            self.history_test_acc.append(test_acc)
            self.history_adv_acc.append(adv_acc)
            evasion_rate = 1 - adv_acc
            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Adversarial accuracy: {adv_acc:.4f} (evasion rate: {evasion_rate:.4f})")

            if adv_acc < 0.8:
                print("Detector too weak – retraining with adversarial samples...")
                self.retrain_detector(adv_samples, self.cfg.loop.retrain_detector_epochs)

            frozen = self.freeze_detector_for_gan()
            anti_w = self.cfg.loop.anti_weight_boost if cycle > 0 else self.cfg.gan.anti_weight
            print("Training GAN against updated detector...")
            self.train_gan(frozen, anti_weight=anti_w)

        print("\n=== Evolution complete ===")
        print("History (test_acc, adv_acc):")
        for i, (t, a) in enumerate(zip(self.history_test_acc, self.history_adv_acc)):
            print(f"Cycle {i}: Test={t:.4f}, Adversarial Acc={a:.4f} (Evasion={1-a:.4f})")
        evolution_df = pd.DataFrame({
            "cycle": range(len(self.history_test_acc)),
            "test_acc": self.history_test_acc,
            "adv_acc": self.history_adv_acc,
            "evasion_rate": [1 - a for a in self.history_adv_acc]
        })
        evolution_df.to_csv("evolution_history.csv", index=False)
        print("Saved evolution history to evolution_history.csv")