from pathlib import Path
from typing import Annotated, List, Optional

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from olp.dataset.data_module import LengthDataModule
from olp.train.callbacks import CustomTQDMProgressBar, InitCallback
from olp.train.plmodel import LengthPredictionModel

app = typer.Typer(help="Train length prediction model")


def validate_file_list(file_list: Optional[List[Path]]) -> Optional[List[Path]]:
    """Validate that files in the file list exist"""
    if file_list:
        for file_path in file_list:
            if not file_path.exists():
                typer.echo(f"Error: File does not exist: {file_path}", err=True)
                raise typer.Exit(1)
            if not file_path.is_file():
                typer.echo(f"Error: Path is not a file: {file_path}", err=True)
                raise typer.Exit(1)
    return file_list


@app.command()
def main(
    model_name: Annotated[
        str, typer.Option(help="Base model name")
    ] = "Qwen/Qwen3-0.6B",
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 1,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 5,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 1e-4,
    reg_weight: Annotated[float, typer.Option(help="Regression loss weight")] = 0.5,
    cls_weight: Annotated[float, typer.Option(help="Classification loss weight")] = 0.5,
    num_workers: Annotated[int, typer.Option(help="Number of data loader workers")] = 4,
    gpus: Annotated[int, typer.Option(help="Number of GPUs")] = 8,
    max_length: Annotated[int, typer.Option(help="Maximum input length")] = 10240,
    file_list: Annotated[
        List[Path], typer.Option(help="List of data files", callback=validate_file_list)
    ] = None,
):
    """Train length prediction model"""
    pl.seed_everything(42)
    if file_list is None:
        file_list = [Path("./output.json")]

    data_module = LengthDataModule(
        data_path="./",
        file_list=file_list,
        tokenizer_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=0.95,
    )

    model = LengthPredictionModel(
        model_name=model_name,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        reg_weight=reg_weight,
        cls_weight=cls_weight,
        max_length=max_length,
    )

    logger = TensorBoardLogger(
        save_dir="./logs", name="length_prediction", version=None
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val/f1_score",
            mode="max",
            save_top_k=5,
            filename="best-{epoch:02d}-{val/f1_score:.2f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        InitCallback(),  # Add initialization callback
        CustomTQDMProgressBar(),  # Add custom progress bar
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=gpus if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        val_check_interval=100,
        strategy="ddp" if gpus > 1 else "auto",
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, data_module)

    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print(f"TensorBoard logs saved at: {logger.log_dir}")
    print("Run the following command to view TensorBoard:")
    print(f"tensorboard --logdir {logger.log_dir}")


if __name__ == "__main__":
    app()
