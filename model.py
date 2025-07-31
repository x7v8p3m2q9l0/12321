import torch
import torch.nn as nn
import torch.nn.functional as F
# Add this to model.py or tokenizer.py
class CharTokenizer:
    def __init__(self, text=None):
        chars = sorted(list(set(text))) if text else []
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    
    def encode(self, s):
        return [self.stoi[ch] for ch in s]
    
    def decode(self, tokens):
        return ''.join(self.itos[i] for i in tokens)

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=512, n_layer=8, n_head=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, idx):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        head_dim = n_embed // n_head
        self.heads = nn.ModuleList([SelfAttention(n_embed, head_dim) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))

class SelfAttention(nn.Module):
    def __init__(self, n_embed, head_dim):
        super().__init__()
        self.q = nn.Linear(n_embed, head_dim, bias=False)
        self.k = nn.Linear(n_embed, head_dim, bias=False)
        self.v = nn.Linear(n_embed, head_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        att = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        att = att.masked_fill(torch.triu(torch.ones(T, T), 1).to(x.device) == 1, float('-inf'))
        att = F.softmax(att, dim=-1)
        return att @ v

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
