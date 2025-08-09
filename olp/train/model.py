import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Linear(hidden_size, 512),
            RMSNorm(512),
            nn.SiLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(256, 64),
            RMSNorm(64),
            nn.SiLU(),
        )
        self.cls_output = nn.Sequential(
            nn.Linear(32, 16),
            RMSNorm(16),
            nn.SiLU(),
            nn.Linear(16, self.num_classes),
        )
        self.reg_output = nn.Sequential(
            nn.Linear(32, 16),
            RMSNorm(16),
            nn.SiLU(),
            nn.Linear(16, self.num_classes),
            RMSNorm(self.num_classes),
            nn.Sigmoid(),
        )

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
        x = F.silu(x[:, :256]) * x[:, 256:]
        x = self.layer_2(x)
        cls_logits = self.cls_output(x[:, : self.num_classes])
        reg_logits = self.reg_output(x[:, self.num_classes :])

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
