import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

NUM_FEATURES = 5    # Represent 5 distinct concepts
HIDDEN_DIM = 2      # Squeeze features into 2 dimensions
SPARSITY = 0.95     # 95% of the time, a feature is 0 (inactive)
NUM_BATCHES = 5000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-2

class AutoEncoder(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()

        # Initialize encoder matrix: maps features to directions in hidden layer (start off with random directions)
        self.W = nn.Parameter(torch.randn(n_features, n_hidden))
        self.b = nn.Parameter(torch.zeros(n_features))

    # Feed forward pass
    def forward(self, x):

        # Normalize rows of W to unit length
        W_normed = self.W / self.W.norm(dim=1, keepdim=True)

        # h = x * W, sends features into hidden layer directions
        hidden = x @ W_normed
        
        # Decoder: reconstruct features from hidden layer. In this toy model, we use the transpose of W (tied weights)
        reconstruction = torch.relu((hidden @ W_normed.T) + self.b)
        return reconstruction, W_normed

def generate_batch(batch_size, n_features, sparsity):

    # Create random feature values (0 to 1)
    data = torch.rand(batch_size, n_features)

    # Apply sparsity mask (set most to 0)
    mask = (torch.rand(batch_size, n_features) > sparsity).float()
    return data * mask

# Training loop
model = AutoEncoder(NUM_FEATURES, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Training on {NUM_FEATURES} features compressed into {HIDDEN_DIM} dimensions...")

loss_history = []

for step in range(NUM_BATCHES):
    # Get a batch of data
    x = generate_batch(BATCH_SIZE, NUM_FEATURES, SPARSITY)
    
    # Forward pass
    x_prime, W_matrix = model(x)
    
    # Loss calculation
    loss = ((x - x_prime)**2).sum(dim=1).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")
        loss_history.append(loss.item())

# Visualization of geometry of superposition
print("Plotting the geometry...")

# Extract the learned weights (the feature vectors)
weights = model.W.detach().numpy()

# Normalize them for plotting
weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)

plt.figure(figsize=(8, 8))

# Plot the unit circle (for reference)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Plot each feature vector
colors = plt.cm.rainbow(np.linspace(0, 1, NUM_FEATURES))
for i in range(NUM_FEATURES):
    plt.arrow(0, 0, weights[i, 0], weights[i, 1], 
              head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i], width=0.01)
    plt.text(weights[i, 0]*1.1, weights[i, 1]*1.1, f"Feature {i+1}", 
             color=colors[i], fontsize=12, ha='center')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title(f"Superposition: {NUM_FEATURES} Features in {HIDDEN_DIM} Dimensions")
plt.grid(True, alpha=0.3)
plt.gca().set_aspect('equal')
plt.show()
print("Done! Check 'superposition_plot.png'. If successful, you should see a pentagon.")