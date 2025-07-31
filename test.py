import torch
from model import SimpleGPT, CharTokenizer

# Trust the tokenizer class
torch.serialization.add_safe_globals({'CharTokenizer': CharTokenizer})

# Load checkpoint
checkpoint = torch.load("gpt_roleplay.pt", map_location="cpu", weights_only=False)
tokenizer = checkpoint["tokenizer"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and move to device
model = SimpleGPT(len(tokenizer.stoi), block_size=128).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# Prepare input
prompt = "Hello"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

# Generate output
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=200)
    print(tokenizer.decode(out[0].tolist()))
