# Token Scope - 输出长度预测模型

基于PyTorch Lightning的输出长度预测模型训练框架，用于预测给定输入的输出文本长度。

## 功能特性

- 🚀 基于PyTorch Lightning的高效训练框架
- 📊 支持回归和分类的组合损失函数
- 🔧 使用Qwen系列模型作为基础编码器
- 📈 集成TensorBoard日志记录
- ⚡ 支持多GPU训练
- 🎯 可配置的超参数调优

## 项目结构

```
token_scope/
├── olp/                    # 主要模块
│   ├── dataset/           # 数据处理模块
│   │   └── data_module.py # PyTorch Lightning数据模块
│   └── train/             # 训练相关模块
│       ├── plmodel.py     # PyTorch Lightning模型
│       ├── model.py       # 核心模型定义
│       ├── loss.py        # 损失函数
│       ├── callbacks.py   # 训练回调
│       └── lr_scheduler.py # 学习率调度器
├── train.py              # 主训练脚本
├── .gitignore           # Git忽略文件
├── .black               # Black代码格式化配置
└── .isort.cfg          # import排序配置
```

## 安装依赖

```bash
pip install pytorch-lightning torch transformers typer tensorboard scikit-learn
```

## 使用方法

### 基本训练

```bash
python train.py
```

### 自定义参数训练

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

### 参数说明

- `--model-name`: 基础模型名称 (默认: "Qwen/Qwen3-0.6B")
- `--batch-size`: 批次大小 (默认: 1)
- `--max-epochs`: 最大训练轮数 (默认: 5)
- `--learning-rate`: 学习率 (默认: 1e-4)
- `--reg-weight`: 回归损失权重 (默认: 0.5)
- `--cls-weight`: 分类损失权重 (默认: 0.5)
- `--num-workers`: 数据加载器工作进程数 (默认: 4)
- `--gpus`: GPU数量 (默认: 8)
- `--max-length`: 最大输入长度 (默认: 10240)
- `--file-list`: 数据文件列表 (默认: ["./output.json"])

## 模型架构

项目使用组合模型架构：
- **编码器**: Qwen系列预训练模型
- **预测头**: 同时进行回归和分类预测
- **损失函数**: 回归损失 + 分类损失的加权组合

## 训练特性

- **混合损失**: 结合MSE回归损失和交叉熵分类损失
- **学习率调度**: 使用余弦退火学习率调度器
- **模型检查点**: 自动保存最佳模型
- **进度监控**: 自定义TQDM进度条和TensorBoard日志

## 数据格式

训练数据应为JSON格式，包含输入文本和对应的输出长度标签。

## 开发工具

项目配置了以下代码质量工具：
- **Black**: Python代码格式化
- **isort**: import语句排序

运行格式化：
```bash
black .
isort .
```

## 监控训练

使用TensorBoard查看训练日志：
```bash
tensorboard --logdir logs
```

## 许可证

[请根据实际情况添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request来改进项目。

---

**English Version**: [README.md](README.md)
