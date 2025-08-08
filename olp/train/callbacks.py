from pytorch_lightning.callbacks import Callback, TQDMProgressBar


class CustomTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super().__init__()
        self.train_progress_bar = None
        self.val_progress_bar = None

    def on_train_start(self, trainer, pl_module):
        """Initialize progress bar when training starts"""
        super().on_train_start(trainer, pl_module)
        print(f"\n{'='*60}")
        print(
            f"Starting training with {trainer.num_training_batches} batches per epoch"
        )
        print(f"Total epochs: {trainer.max_epochs}")
        print(f"Batch size: {trainer.datamodule.batch_size}")
        print(f"GPU devices: {trainer.num_devices}")
        print(f"{'='*60}\n")

    def on_train_epoch_start(self, trainer, pl_module):
        """Display information at the start of each epoch"""
        super().on_train_epoch_start(trainer, pl_module)
        print(f"\n{'='*40}")
        print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
        print(f"{'='*40}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Display detailed information at the end of each batch"""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        # Display detailed information every 50 batches
        if batch_idx % 50 == 0:
            # Get current learning rate
            lr = trainer.optimizers[0].param_groups[0]["lr"]

            # Get loss information
            if outputs is not None:
                if isinstance(outputs, dict):
                    total_loss = outputs.get("loss", 0)
                else:
                    total_loss = outputs

                # Get training metrics
                train_reg_loss = trainer.callback_metrics.get("train/reg_loss", 0)
                train_cls_loss = trainer.callback_metrics.get("train/cls_loss", 0)
                train_cls_acc = trainer.callback_metrics.get("train/cls_acc", 0)
                train_f1 = trainer.callback_metrics.get("train/f1_score", 0)

                print(
                    f"\nBatch {batch_idx}/{trainer.num_training_batches} | "
                    f"Loss: {total_loss:.4f} | Reg: {train_reg_loss:.4f} | "
                    f"Cls: {train_cls_loss:.4f} | Acc: {train_cls_acc:.3f} | "
                    f"F1: {train_f1:.3f} | LR: {lr:.2e}"
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        """Display information when validation starts"""
        super().on_validation_epoch_start(trainer, pl_module)
        print(f"\n{'='*30}")
        print(f"Validation Epoch {trainer.current_epoch + 1}")
        print(f"{'='*30}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Display summary information when validation ends"""
        super().on_validation_epoch_end(trainer, pl_module)

        # Get validation metrics
        val_loss = trainer.callback_metrics.get("val/total_loss", 0)
        val_cls_acc = trainer.callback_metrics.get("val/cls_acc", 0)
        val_reg_loss = trainer.callback_metrics.get("val/reg_loss", 0)
        val_f1 = trainer.callback_metrics.get("val/f1_score", 0)

        print(f"\nValidation Results:")
        print(f"  Total Loss: {val_loss:.4f}")
        print(f"  Classification Accuracy: {val_cls_acc:.3f}")
        print(f"  F1 Score: {val_f1:.3f}")
        print(f"  Regression Loss: {val_reg_loss:.4f}")
        print(f"{'='*30}\n")


class InitCallback(Callback):
    """Callback to ensure model parameters are properly initialized before training starts"""

    def on_fit_start(self, trainer, pl_module):
        """Called when training starts"""
        print("Ensuring UnifiedHead parameters are properly initialized...")
        if hasattr(pl_module, "unified_head"):
            pl_module.unified_head._init_weights()
            print("UnifiedHead parameters re-initialized successfully!")

        # Print some parameter statistics
        total_params = sum(p.numel() for p in pl_module.unified_head.parameters())
        trainable_params = sum(
            p.numel() for p in pl_module.unified_head.parameters() if p.requires_grad
        )
        print(
            f"UnifiedHead - Total parameters: {total_params:,}, Trainable: {trainable_params:,}"
        )

        # Print learning rate scheduler information
        if hasattr(pl_module, "configure_optimizers"):
            optimizers = pl_module.configure_optimizers()
            if "lr_scheduler" in optimizers:
                scheduler = optimizers["lr_scheduler"]["scheduler"]
                if hasattr(scheduler, "steps_per_epoch"):
                    print(
                        f"Learning rate scheduler: EpochCosineAnnealingLR with {scheduler.steps_per_epoch} steps per epoch"
                    )
                    print(f"Base learning rates: {scheduler.base_lrs}")
                    print(f"Initial learning rate: {scheduler.get_lr()[0]:.2e}")
