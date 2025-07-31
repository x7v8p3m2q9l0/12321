import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from model import SimpleGPT
import random

block_size = 128

# === Tokenizer ===
class CharTokenizer:
    def __init__(self, dataset):
        text = " ".join(sample["conversations"][0]["value"] + sample["conversations"][1]["value"]
                        for sample in dataset if len(sample["conversations"]) >= 2)
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

# === Dataset ===
class RoleplayDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.samples = []
        for sample in dataset:
            conv = sample["conversations"]
            if len(conv) >= 2 and conv[0]["from"] == "user" and conv[1]["from"] == "assistant":
                prompt = conv[0]["value"]
                reply = conv[1]["value"]
                text = prompt + "\n" + reply
                tokens = tokenizer.encode(text)
                if len(tokens) > 0:
                    self.samples.append(tokens)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx][:block_size+1]
        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)
        return x, y

# === Train ===
def train():
    raw = load_dataset("OdiaGenAI/roleplay_english")["train"]
    tokenizer = CharTokenizer(raw)
    dataset = RoleplayDataset(raw, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleGPT(tokenizer.vocab_size, block_size).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epoch=1
    streaks=0
    try:
        while True:
            pbar = tqdm(loader)
            for x, y in pbar:
                x, y = x.to(model.device), y.to(model.device)
                logits = model(x)
                B, T, C = logits.shape
                loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss=loss.item()
                pbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
            epoch+=1
            if loss<=0.25:
                streaks+=1
                if streaks > 10:
                    break
    except KeyboardInterrupt:
        print("Skibidi")
    finally:
        torch.save({"model": model.state_dict(), "tokenizer": tokenizer}, "gpt_roleplay.pt")
        print("Model saved to gpt_roleplay.pt")

if __name__ == "__main__":
    train()
