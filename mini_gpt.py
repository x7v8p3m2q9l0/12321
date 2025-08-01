import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb
import argparse

@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocab size
    n_layer: int = 24        # Number of transformer layers
    n_head: int = 20         # Number of attention heads
    n_embd: int = 1280       # Embedding dimension
    block_size: int = 2048   # Max sequence length
    
    # Regularization
    dropout: float = 0.1
    bias: bool = True        # Use bias in linear layers
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = True

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.dropout == 0.0:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class HermesDataset(Dataset):
    def __init__(self, tokenizer, block_size=1024, split='train'):
        print(f"Loading Hermes-3 dataset ({split})...")
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load dataset
        dataset = load_dataset("NousResearch/Hermes-3-Dataset", split=split)
        
        # Process conversations
        self.examples = []
        print("Processing conversations...")
        
        for item in tqdm(dataset):
            conversation = item['conversations']
            
            # Format conversation
            text = ""
            for msg in conversation:
                role = msg['from']
                content = msg['value']
                
                if role == 'system':
                    text += f"<|system|>\n{content}\n"
                elif role == 'human':
                    text += f"<|user|>\n{content}\n"
                elif role == 'gpt':
                    text += f"<|assistant|>\n{content}\n"
            
            text += "<|endoftext|>"
            
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Split into chunks
            for i in range(0, len(tokens) - block_size + 1, block_size):
                chunk = tokens[i:i + block_size]
                if len(chunk) == block_size:
                    self.examples.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class GPTTrainer:
    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
        # Setup optimizer
        self.optimizer = self.configure_optimizers()
        
        # Setup datasets
        print("Preparing datasets...")
        self.train_dataset = HermesDataset(tokenizer, config.block_size, 'train')
        
        # Use a subset for validation to save memory
        print("Loading validation data...")
        val_dataset_full = load_dataset("NousResearch/Hermes-3-Dataset", split='train')
        val_size = min(1000, len(val_dataset_full))
        val_indices = np.random.choice(len(val_dataset_full), val_size, replace=False)
        val_subset = val_dataset_full.select(val_indices)
        
        # Process validation data
        self.val_examples = []
        for item in tqdm(val_subset, desc="Processing validation"):
            conversation = item['conversations']
            text = ""
            for msg in conversation:
                role = msg['from']
                content = msg['value']
                if role == 'system':
                    text += f"<|system|>\n{content}\n"
                elif role == 'human':
                    text += f"<|user|>\n{content}\n"
                elif role == 'gpt':
                    text += f"<|assistant|>\n{content}\n"
            text += "<|endoftext|>"
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) >= config.block_size:
                chunk = tokens[:config.block_size]
                self.val_examples.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"Validation examples: {len(self.val_examples)}")
        
        # Setup data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=8,  # Adjust based on your GPU memory
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def configure_optimizers(self):
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        
        # Separate parameters into decay and no_decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in param_dict.items():
            # Remove any _orig_mod prefix from compiled models
            clean_name = name.replace('_orig_mod.', '')
            
            if param.dim() >= 2:  # Weights (2D tensors and above)
                if 'ln_' in clean_name or 'layernorm' in clean_name.lower() or clean_name.endswith('.bias'):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            else:  # Biases and 1D parameters
                no_decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        print(f"Optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no decay params")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        return optimizer

    def get_lr(self):
        if self.iter_num < self.config.warmup_iters:
            return self.config.learning_rate * self.iter_num / self.config.warmup_iters
        if self.iter_num > self.config.lr_decay_iters:
            return self.config.min_lr
        
        decay_ratio = (self.iter_num - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = []
        
        for i in range(min(100, len(self.val_examples))):
            x = self.val_examples[i].unsqueeze(0).to(self.config.device)
            y = x.clone()
            
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16 if self.config.dtype == 'bfloat16' else torch.float16):
                logits, loss = self.model(x, y)
            
            losses.append(loss.item())
        
        self.model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {path}")

    def train(self, max_iters=100000, eval_interval=500, save_interval=2000):
        print(f"Training for {max_iters} iterations...")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        self.model.train()
        running_loss = 0.0
        
        # Training loop
        for batch_idx, batch in enumerate(self.train_loader):
            if self.iter_num >= max_iters:
                break
                
            # Learning rate schedule
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            x = batch.to(self.config.device)
            y = x.clone()
            
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16 if self.config.dtype == 'bfloat16' else torch.float16):
                logits, loss = self.model(x, y)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if self.config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Logging
            if self.iter_num % 100 == 0:
                avg_loss = running_loss / 100
                print(f"iter {self.iter_num}: loss {avg_loss:.4f}, lr {lr:.2e}")
                running_loss = 0.0
            
            # Evaluation
            if self.iter_num % eval_interval == 0 and self.iter_num > 0:
                val_loss = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pt")
            
            # Save checkpoint
            if self.iter_num % save_interval == 0 and self.iter_num > 0:
                self.save_checkpoint(f"checkpoint_{self.iter_num}.pt")
            
            self.iter_num += 1

    def chat(self, prompt, max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.9):
        """Interactive chat with the model"""
        self.model.eval()
        
        # Format prompt
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize
        tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        x = torch.tensor(tokens, dtype=torch.long, device=self.config.device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16 if self.config.dtype == 'bfloat16' else torch.float16):
                y = self.model.generate(x, max_new_tokens, temperature, top_k, top_p)
        
        # Decode
        generated_tokens = y[0][len(tokens):].tolist()
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        self.model.train()
        return response.strip()

def create_mini_gpt():
    config = GPTConfig(
        vocab_size=50257,
        n_layer=24,        # 24 layers
        n_head=20,         # 20 attention heads  
        n_embd=1280,       # 1280 embedding dimension
        block_size=2048,   # 2048 context length
        dropout=0.1,
    )
    
    model = GPT(config)
    return model, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train', choices=['train', 'chat', 'eval'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    # Create model and tokenizer
    print("Initializing model and tokenizer...")
    model, config = create_mini_gpt()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Add special tokens for chat format
    special_tokens = ['<|system|>', '<|user|>', '<|assistant|>', '<|endoftext|>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    # Resize model embeddings
    config.vocab_size = len(tokenizer)
    model.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
    model.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    model.transformer.wte.weight = model.lm_head.weight
    
    # Move to device
    model = model.to(config.device)
    
    print(f"Model created with {model.get_num_params():,} parameters")
    print(f"Using device: {config.device}")

    # Create trainer (before compilation to avoid parameter name issues)
    trainer = GPTTrainer(model, config, tokenizer)
    
    # Compile model for faster training (PyTorch 2.0+) - after trainer creation
    if config.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        trainer.model = torch.compile(trainer.model)

    # Create trainer (before compilation to avoid parameter name issues)
    trainer = GPTTrainer(model, config, tokenizer)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.action == 'train':
        print("Starting training...")
        trainer.train(max_iters=args.max_iters)
        
    elif args.action == 'chat':
        print("Starting interactive chat...")
        print("Type 'quit' to exit")
        
        while True:
            prompt = input("\nYou: ")
            if prompt.lower() == 'quit':
                break
                
            response = trainer.chat(prompt)
            print(f"Assistant: {response}")
            print("Evaluating model...")
            val_loss = trainer.estimate_loss()
            print(f"Validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()