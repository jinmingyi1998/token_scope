import torch
import torch.nn as nn
from transformers import AutoModel


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        x = x.float()  # x: [batch_size, hidden_size]
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight.float() * x
        return x


class UnifiedHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 6, max_length: int = 10240):
        super().__init__()
        self.max_length = max_length
        self.num_classes = num_classes

        self.layer_1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            RMSNorm(256),
            nn.SiLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(256, 64),
            RMSNorm(64),
            nn.SiLU(),
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(64, 16),
            RMSNorm(16),
            nn.SiLU(),
        )

        self.cls_output = nn.Linear(16, num_classes)
        self.reg_output = nn.Linear(16, num_classes)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Randomly initialize all weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module == self.cls_output:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                elif module == self.reg_output:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    # Make initial predictions in the linear region of sigmoid
                    nn.init.zeros_(module.bias)  # sigmoid(0) = 0.5
                else:
                    nn.init.xavier_uniform_(module.weight)

                # Unified bias handling
                if module.bias is not None and module != self.reg_output:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, RMSNorm):
                if hasattr(module, "weight"):
                    nn.init.ones_(module.weight)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, hidden_size]
        x = self.layer_1(hidden_states)
        x = self.layer_2(x)
        x = self.layer_3(x)
        cls_logits = self.cls_output(x)  # [batch_size, num_classes]
        reg_logits = self.sigmoid(self.reg_output(x))
        return reg_logits, cls_logits


class OutputLengthPredictor(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", num_classes: int = 7):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.head = UnifiedHead(
            self.base_model.config.hidden_size,
            num_classes=num_classes,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        last_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        reg_logits, cls_logits = self.head(last_hidden_states)
        return reg_logits, cls_logits
