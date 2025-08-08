import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class EpochCosineAnnealingLR(CosineAnnealingLR):
    """Cosine annealing learning rate scheduler that restarts every epoch"""

    def __init__(self, optimizer, steps_per_epoch, **kwargs):
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0
        super().__init__(optimizer, T_max=steps_per_epoch, **kwargs)

        if not hasattr(self, "base_lrs"):
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_step // self.steps_per_epoch
        step_in_epoch = self.current_step % self.steps_per_epoch
        for i, param_group in enumerate(self.optimizer.param_groups):
            if "lr" in param_group:
                eta_min = param_group.get("eta_min", 0)
                eta_max = (
                    self.base_lrs[i] if i < len(self.base_lrs) else param_group["lr"]
                )

                lr = (
                    eta_min
                    + (eta_max - eta_min)
                    * (1 + np.cos(np.pi * step_in_epoch / self.steps_per_epoch))
                    / 2
                )
                param_group["lr"] = lr

        self.current_step += 1

    def get_lr(self):
        step_in_epoch = self.current_step % self.steps_per_epoch
        lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            eta_min = param_group.get("eta_min", 0)
            eta_max = self.base_lrs[i] if i < len(self.base_lrs) else param_group["lr"]
            lr = (
                eta_min
                + (eta_max - eta_min)
                * (1 + np.cos(np.pi * step_in_epoch / self.steps_per_epoch))
                / 2
            )
            lrs.append(lr)
        return lrs
