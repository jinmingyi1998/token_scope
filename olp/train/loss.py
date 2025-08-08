import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, cls_weight=0.5, reg_weight=0.5, ce_class_weights=None):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        if ce_class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
            print(
                f"Initialized CrossEntropyLoss with class weights: {ce_class_weights}"
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            print("Initialized CrossEntropyLoss without weights")
        self.l2_loss = nn.MSELoss()

    def forward(
        self, cls_logits, reg_logits, cls_label, cls_soft_target, global_step: int = 0
    ):
        cls_loss = self.ce_loss(cls_logits, cls_label)
        cls_prob = F.softmax(cls_logits, dim=-1)
        all_reg_loss = self.l2_loss(reg_logits, cls_soft_target) * 0.1
        single_reg_loss = self.l2_loss(reg_logits * cls_prob, cls_soft_target) * 0.9
        reg_loss = all_reg_loss + single_reg_loss

        total_loss = self.cls_weight * cls_loss + reg_loss * (
            self.reg_weight if global_step > 100 else 0
        )
        return total_loss, cls_loss, reg_loss
