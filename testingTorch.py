import torch
import torch.nn as nn
import time

# 1. Setup Device (Will use GPU if you have one, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Scale up the dimensions (Adjust these to push your hardware)
# 10,000 x 10,000 matrices require significant RAM/VRAM
INPUT_SIZE = 8192
HIDDEN_SIZE = 8192
OUTPUT_SIZE = 4096
BATCH_SIZE = 128  # Number of samples processed at once

class HeavyModel(nn.Module):
    def __init__(self):
        super(HeavyModel, self).__init__()
        # Large Linear layers (Matrix Multiplications)
        self.layer1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Heavy Matrix Ops + Summing Biases
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return self.softmax(x)

# 3. Initialize Model and Large Random Data
model = HeavyModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Create massive random tensors
X = torch.randn(BATCH_SIZE, INPUT_SIZE).to(device)
y = torch.randint(0, OUTPUT_SIZE, (BATCH_SIZE,)).to(device)

print("Starting heavy computation loop...")
start_time = time.time()

# 4. The Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward Pass: Includes Matrix Multiplications, ReLU, and Softmax
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward Pass: Computes gradients for millions of parameters
    loss.backward()
    
    # Parameter Adjustment
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Time Elapsed: {time.time() - start_time:.2f}s")

print(f"\nTotal computation time: {time.time() - start_time:.2f} seconds")
