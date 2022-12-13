import torch
import torch.nn as nn
from transformers import BertModel

class GlobalPointer(nn.Module):
    def __init__(self, config, num_type):
        super(GlobalPointer, self).__init__()
        self.config = config
        # self.RoPE = self.config.RoPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bert_dim = self.config.bert_dim
        self.num_type = num_type
        self.linear = nn.Linear(self.bert_dim, self.num_type * self.config.hidden_size * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, bert_last_hidden_state, attention_mask):
        # input_ids = data["input_ids"].to(self.device)
        # attention_mask = data["mask"].to(self.device)

        # [batch_size, seq_len, bert_dim]
        last_hidden_state = bert_last_hidden_state[0]
        batch_size = last_hidden_state.size(0)
        seq_len = last_hidden_state.size(1)

        # [batch_size, seq_len, num_type * hidden_size * 2]
        outputs = self.linear(last_hidden_state)
        # [[batch_size, seq_len, hidden_size * 2], [batch_size, seq_len, hidden_size * 2], [batch_size, seq_len, hidden_size * 2]]
        outputs = torch.split(outputs, self.config.hidden_size * 2, dim=-1)
        # [batch_size, seq_len, num_type, hidden_size * 2]
        outputs = torch.stack(outputs, dim=-2)
        # [batch_size, seq_len, num_type, hidden_size]
        qw, kw = outputs[..., :self.config.hidden_size], outputs[..., self.config.hidden_size:]

        if self.config.RoPE:
            # [batch_size, seq_len, hidden_size]
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.config.hidden_size)

            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # [batch_size, num_type, seq_len, seq_len]
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # [batch_size, num_type, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_type, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12

        return logits / self.config.hidden_size ** 0.5

class GlobalLinker(nn.Module):
    def __init__(self, config):
        super(GlobalLinker, self).__init__()
        self.config =config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.relation_model = GlobalPointer(self.config, 2)
        self.head_model = GlobalPointer(self.config, self.config.num_rel)
        self.tail_model = GlobalPointer(self.config, self.config.num_rel)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        hidden_state = self.bert(input_ids, attention_mask=attention_mask)
        relation_logits = self.relation_model(hidden_state, attention_mask)
        head_logits = self.head_model(hidden_state, attention_mask)
        tail_logits = self.tail_model(hidden_state, attention_mask)
        return relation_logits, head_logits, tail_logits

