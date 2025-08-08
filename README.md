# Token Scope - Output Length Prediction Model

A PyTorch Lightning-based training framework for output length prediction models, designed to predict the length of output text given input text.

## Features

- 🚀 Efficient training framework based on PyTorch Lightning
- 📊 Combined regression and classification loss functions
- 🔧 Qwen series models as base encoders
- 📈 Integrated TensorBoard logging
- ⚡ Multi-GPU training support
- 🎯 Configurable hyperparameter tuning

## Project Structure

```
token_scope/
├── olp/                    # Main module
│   ├── dataset/           # Data processing module
│   │   └── data_module.py # PyTorch Lightning data module
│   └── train/             # Training related modules
│       ├── plmodel.py     # PyTorch Lightning model
│       ├── model.py       # Core model definition
│       ├── loss.py        # Loss functions
│       ├── callbacks.py   # Training callbacks
│       └── lr_scheduler.py # Learning rate scheduler
├── train.py              # Main training script
├── .gitignore           # Git ignore file
├── .black               # Black code formatting config
└── .isort.cfg          # Import sorting config
```

## Installation

```bash
pip install pytorch-lightning torch transformers typer tensorboard scikit-learn
```

## Usage

### Basic Training

```bash
python train.py
```

### Custom Parameter Training

```bash
python train.py \
    --model-name "Qwen/Qwen3-0.6B" \
    --batch-size 4 \
    --max-epochs 10 \
    --learning-rate 2e-4 \
    --reg-weight 0.6 \
    --cls-weight 0.4 \
    --gpus 4 \
    --max-length 8192 \
    --file-list data1.json data2.json
```

### Parameters

- `--model-name`: Base model name (default: "Qwen/Qwen3-0.6B")
- `--batch-size`: Batch size (default: 1)
- `--max-epochs`: Maximum training epochs (default: 5)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--reg-weight`: Regression loss weight (default: 0.5)
- `--cls-weight`: Classification loss weight (default: 0.5)
- `--num-workers`: Number of data loader workers (default: 4)
- `--gpus`: Number of GPUs (default: 8)
- `--max-length`: Maximum input length (default: 10240)
- `--file-list`: List of data files (default: ["./output.json"])

## Model Architecture

The project uses a combined model architecture:
- **Encoder**: Qwen series pre-trained models
- **Prediction Head**: Simultaneous regression and classification prediction
- **Loss Function**: Weighted combination of regression and classification losses

## Training Features

- **Mixed Loss**: Combines MSE regression loss and cross-entropy classification loss
- **Learning Rate Scheduling**: Uses cosine annealing learning rate scheduler
- **Model Checkpointing**: Automatically saves best models
- **Progress Monitoring**: Custom TQDM progress bar and TensorBoard logging

## Data Format

Training data should be in JSON format containing input text and corresponding output length labels.

## Development Tools

The project is configured with the following code quality tools:
- **Black**: Python code formatting
- **isort**: Import statement sorting

Run formatting:
```bash
black .
isort .
```

## Training Monitoring

Use TensorBoard to view training logs:
```bash
tensorboard --logdir logs
```

## License

[Please add license information as appropriate]

## Contributing

Issues and Pull Requests are welcome to improve the project.

---

**中文版本**: [README_CN.md](README_CN.md)
