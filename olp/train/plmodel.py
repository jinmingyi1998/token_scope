import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam

from olp.dataset.data_module import LengthDataset
from olp.train.loss import CombinedLoss
from olp.train.lr_scheduler import EpochCosineAnnealingLR
from olp.train.model import OutputLengthPredictor


class LengthPredictionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        learning_rate: float = 1e-4,
        max_epochs: int = 5,
        reg_weight: float = 0.5,
        cls_weight: float = 0.5,
        max_length: int = 10240,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = OutputLengthPredictor(model_name=model_name, num_classes=7)
        self.loss = None

        # Hyperparameters
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.max_length = max_length

        max_class_id = LengthDataset.cls_boundaries[-1]["cls_id"]

        self.start_mapping = torch.zeros(max_class_id + 1)
        self.end_mapping = torch.zeros(max_class_id + 1)
        self.range_mapping = torch.zeros(max_class_id + 1)

        for boundary in LengthDataset.cls_boundaries:
            self.start_mapping[boundary["cls_id"]] = boundary["start"]
            self.end_mapping[boundary["cls_id"]] = boundary["end"]
            self.range_mapping[boundary["cls_id"]] = boundary["range"]

    def setup(self, stage: str = None):
        """Setup model components"""
        if stage == "fit" and self.ce_loss is None:
            # Move mapping tensors to the correct device
            self.start_mapping = self.start_mapping.to(self.device)
            self.end_mapping = self.end_mapping.to(self.device)
            self.range_mapping = self.range_mapping.to(self.device)

            # Get class weights from data module
            class_weights = None
            if hasattr(self.trainer, "datamodule") and hasattr(
                self.trainer.datamodule, "class_weights"
            ):
                class_weights = torch.tensor(
                    self.trainer.datamodule.class_weights, device=self.device
                )
            self.loss = CombinedLoss(
                cls_weight=self.cls_weight,
                reg_weight=self.reg_weight,
                ce_class_weights=class_weights,
            )

    def forward(self, input_ids, attention_mask):
        # Get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Get last token hidden states
        last_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # Pass through unified head
        reg_logits, cls_logits = self.unified_head(last_hidden_states)
        return cls_logits, reg_logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cls_label = batch["cls_label"]
        cls_soft_target = batch["cls_soft_target"]

        cls_logits, reg_logits = self(input_ids, attention_mask)
        # cls_logits: [batch_size, num_classes]
        # reg_logits: [batch_size, num_classes]

        # Calculate classification metrics
        cls_pred = torch.argmax(cls_logits, dim=1)
        cls_acc = (cls_pred == cls_label).float().mean()

        # Calculate F1 score
        f1 = f1_score(
            cls_label.cpu().numpy(), cls_pred.cpu().numpy(), average="weighted"
        )
        global_step = getattr(self.trainer, "global_step", -1)
        total_loss, cls_loss, reg_loss = self.loss(
            cls_logits, reg_logits, cls_label, cls_soft_target, global_step
        )

        # Log metrics
        self.log("train/reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log("train/f1_score", f1, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/cls_acc", cls_acc, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        true_length = batch["length"]
        cls_label = batch["cls_label"]

        # Forward pass
        reg_logits, cls_logits = self(input_ids, attention_mask)
        # pred_length: [batch_size], cls_logits: [batch_size, num_classes]
        # Calculate classification metrics
        cls_pred = torch.argmax(cls_logits, dim=1)
        # Ensure cls_label is on the correct device
        cls_label_device = cls_label.to(cls_logits.device)
        cls_acc = (cls_pred == cls_label_device).float().mean()

        # Calculate F1 score
        f1 = f1_score(
            cls_label_device.cpu().numpy(), cls_pred.cpu().numpy(), average="weighted"
        )

        device = cls_logits.device
        length_range = self.range_mapping[cls_pred].to(device)  # [batch_size]
        start_values = self.start_mapping[cls_pred].to(device)  # [batch_size]

        # Correct indexing: select corresponding class values from each sample's reg_logits
        batch_indices = torch.arange(cls_pred.size(0), device=device)
        selected_reg_logits = reg_logits[batch_indices, cls_pred]  # [batch_size]

        # Calculate predicted length
        pred_length = start_values + length_range * selected_reg_logits
        length_loss = torch.abs(pred_length - true_length).mean()

        self.log("val/cls_acc", cls_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1_score", f1, on_epoch=True, prog_bar=True)
        self.log("val/length_loss", length_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Adam optimizer
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        # Calculate steps per epoch
        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.max_epochs
            if hasattr(self.trainer, "estimated_stepping_batches")
            else 3e4
        )

        # Use custom EpochCosineAnnealingLR scheduler
        scheduler = EpochCosineAnnealingLR(optimizer, steps_per_epoch=steps_per_epoch)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
                "monitor": None,  # Don't monitor validation metrics
            },
        }
