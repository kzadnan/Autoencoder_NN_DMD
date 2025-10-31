import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pydmd import DMD

def create_timestep_matrix(file_path, output_file="timestep_matrix.txt"):
    """
    Reads atomic positions from a LAMMPS trajectory file and converts them into a matrix.
    
    Stores:
    - Atom ID, Type, x, y, z in rows
    - Timesteps as columns
    - Box bounds separately
    
    Returns:
    - matrix: NumPy array (num_atoms * 5, timesteps)
    - pos_matrix: NumPy array (num_atoms * 3, timesteps) for x, y, z only
    - sorted_timesteps: List of timesteps
    - box_bounds: Dictionary of box bounds for each timestep
    - num_atoms: Number of atoms detected
    """
    timestep_data = {}
    box_bounds = {}
    current_timestep = None
    current_positions = []
    num_atoms = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "ITEM: TIMESTEP" in line:
                if current_positions and current_timestep is not None:
                    timestep_data[current_timestep] = np.array(current_positions).flatten()
                    current_positions = []
                i += 1
                current_timestep = int(lines[i].strip())
            elif "ITEM: NUMBER OF ATOMS" in line:
                i += 1
                num_atoms = int(lines[i].strip())
            elif "ITEM: BOX BOUNDS" in line:
                i += 1
                x_bounds = list(map(float, lines[i].strip().split()))
                i += 1
                y_bounds = list(map(float, lines[i].strip().split()))
                i += 1
                z_bounds = list(map(float, lines[i].strip().split()))
                box_bounds[current_timestep] = (x_bounds, y_bounds, z_bounds)
            elif "ITEM: ATOMS id type xs ys zs" in line:
                i += 1
                for _ in range(num_atoms):
                    values = list(map(float, lines[i].strip().split()))
                    current_positions.extend(values)
                    i += 1
                continue
            i += 1

    if current_positions and current_timestep is not None:
        timestep_data[current_timestep] = np.array(current_positions).flatten()

    sorted_timesteps = sorted(timestep_data.keys())
    num_rows = num_atoms * 5
    num_cols = len(sorted_timesteps)
    matrix = np.zeros((num_rows, num_cols))
    for j, timestep in enumerate(sorted_timesteps):
        matrix[:, j] = timestep_data[timestep]

    # Extract positions (x, y, z)
    indices = []
    for i in range(num_atoms):
        indices.extend([5*i + 2, 5*i + 3, 5*i + 4])  # x, y, z for each atom
    pos_matrix = matrix[indices, :]

    print(f"Matrix shape: {matrix.shape}, Position matrix shape: {pos_matrix.shape}")
    return matrix, pos_matrix, sorted_timesteps, box_bounds, num_atoms

def robust_normalize(matrix):
    """
    Normalize matrix, skipping rows with zero standard deviation.
    
    Returns:
    - normalized_matrix: Normalized data
    - valid_indices: Indices of rows with non-zero std
    """
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    valid_indices = np.where(std.flatten() > 1e-10)[0]
    normalized_matrix = matrix.copy()
    normalized_matrix[valid_indices, :] = (matrix[valid_indices, :] - mean[valid_indices, :]) / std[valid_indices, :]
    normalized_matrix[~np.isin(np.arange(matrix.shape[0]), valid_indices), :] = 0
    if np.any(np.isnan(normalized_matrix)):
        print("Warning: NaN values detected after normalization")
        normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)
    return normalized_matrix, valid_indices

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def train_autoencoder(model, X_train, X_val, epochs=50, lr=0.0001, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

# Load data
matrix, pos_matrix, sorted_timesteps, box_bounds, num_atoms = create_timestep_matrix("nvt.tj")

# Check for NaN in raw data
if np.any(np.isnan(pos_matrix)):
    print("Error: NaN values in pos_matrix")
    pos_matrix = np.nan_to_num(pos_matrix, nan=0.0)

# Robust normalization
pos_matrix, valid_indices = robust_normalize(pos_matrix)
print(f"Valid indices shape: {valid_indices.shape}")

# Prepare for autoencoder
X = pos_matrix.T  # Shape: (T, 3 * num_atoms)
X_tensor = torch.FloatTensor(X)
train_size = int(0.8 * len(X_tensor))
X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]

# Train autoencoder
input_dim = pos_matrix.shape[0]
autoencoder = Autoencoder(input_dim=input_dim)
train_losses, val_losses = train_autoencoder(autoencoder, X_train, X_val)

# Get latent representations
autoencoder.eval()
with torch.no_grad():
    _, Z = autoencoder(X_tensor)
Z = Z.numpy().T  # Shape: (latent_dim, T)

# Check latent representations
if np.any(np.isnan(Z)):
    print("Error: NaN values in latent representations")
    Z = np.nan_to_num(Z, nan=0.0)

# Apply DMD
dmd = DMD(svd_rank=10, exact=True)
try:
    dmd.fit(Z.T)  # Shape: (T, latent_dim)
except np.linalg.LinAlgError as e:
    print(f"DMD failed: {e}")
    exit()

# Reconstruct data
recon_data = np.zeros((X.shape[0], X.shape[1]))  # Shape: (T, 3 * num_atoms)
with torch.no_grad():
    # Take real part to handle complex DMD output
    latent_recon = np.real(dmd.reconstructed_data)  # Shape: (T, latent_dim)
    recon_tensor = autoencoder.decoder(torch.FloatTensor(latent_recon)).detach().numpy()
    recon_data[:, :] = recon_tensor[:, :]  # Ensure correct shape

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Autoencoder Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(X[:, valid_indices[0]], label='True X (first valid atom)')
plt.plot(recon_data[:, valid_indices[0]], label='Reconstructed X (first valid atom)')
plt.title('DMD Reconstruction in Original Space')
plt.xlabel('Timestep')
plt.ylabel('X Position')
plt.legend()

plt.tight_layout()
plt.savefig('autoencoder_dmd_results.png')
plt.show()