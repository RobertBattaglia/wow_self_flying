import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import matplotlib.pyplot as plt


class MinimapDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # Normalize image to [0,1] and convert to tensor
        frame = self.frames[idx].astype(np.float32) / 255.0
        frame = torch.tensor(frame).unsqueeze(0)  # Add channel dimension
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return frame, label


class MinimapCNN(nn.Module):
    def __init__(self, num_actions=4):
        super(MinimapCNN, self).__init__()
        
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate output dimension after convolutions
        # For typical 200x200 input, this will be 128 x 11 x 11
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions),
            nn.Sigmoid()  # For multi-label classification (multiple keys can be pressed at once)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_and_balance_data():
    """Load data from batch files and balance it."""
    print("Loading data from batch files...")
    
    # Get all batch files
    batch_files = glob.glob("data/batches/batch_*.npz")
    
    if len(batch_files) == 0:
        raise ValueError("No batch files found in data/batches/")
    
    # Load and combine all batches
    all_frames = []
    all_keys = []
    
    for batch_file in batch_files:
        data = np.load(batch_file)
        all_frames.append(data['input_data'])
        all_keys.append(data['output_data'])
    
    all_frames = np.vstack(all_frames)
    all_keys = np.vstack(all_keys)
    
    print(f"Loaded {len(all_frames)} total frames")
    print(f"Keys distribution: {np.sum(all_keys, axis=0)}")
    
    # Create a permutation for shuffling
    np.random.seed(42)  # For reproducibility
    permutation = np.random.permutation(len(all_frames))
    all_frames = all_frames[permutation]
    all_keys = all_keys[permutation]
    
    # Balance the data
    # Create class weights for each action type
    action_counts = np.sum(all_keys, axis=0)
    max_count = np.max(action_counts)
    class_weights = max_count / (action_counts + 1e-6)  # Avoid division by zero
    
    # Calculate sample weights
    sample_weights = np.ones(len(all_keys))
    for i in range(len(all_keys)):
        for j in range(len(class_weights)):
            if all_keys[i][j] > 0:
                sample_weights[i] = class_weights[j]
                break
    
    # Calculate split index for train/val (80/20 split)
    split_idx = int(len(all_frames) * 0.8)
    frames_train = all_frames[:split_idx]
    keys_train = all_keys[:split_idx]
    frames_val = all_frames[split_idx:]
    keys_val = all_keys[split_idx:]
    
    return frames_train, keys_train, frames_val, keys_val, sample_weights


def train_model(model, train_loader, val_loader, device, num_epochs=30):
    """Train the model."""
    print(f"Training model on {device}...")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for frames, keys in train_loader:
            frames, keys = frames.to(device), keys.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, keys)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * frames.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, keys in val_loader:
                frames, keys = frames.to(device), keys.to(device)
                outputs = model(frames)
                loss = criterion(outputs, keys)
                val_loss += loss.item() * frames.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Adjust learning rate
        scheduler.step(val_loss)
    
    return model, history


def plot_training_history(history, timestamp):
    """Plot training/validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/training_history-{timestamp}.png")
    plt.show()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and balance data
    frames_train, keys_train, frames_val, keys_val, sample_weights = load_and_balance_data()
    
    # Create datasets
    train_dataset = MinimapDataset(frames_train, keys_train)
    val_dataset = MinimapDataset(frames_val, keys_val)
    
    # Create sampler for balancing
    sampler = WeightedRandomSampler(
        weights=sample_weights[:len(train_dataset)],
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = MinimapCNN().to(device)
    print(model)
    
    # Train model
    trained_model, history = train_model(model, train_loader, val_loader, device, num_epochs=30)
    
    timestamp = int(time.time())
    # Plot training history
    plot_training_history(history, timestamp)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_filename = f"minimap_cnn_{timestamp}.pth"
    model_path = os.path.join("models", model_filename)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Create/update symlink for latest model
    latest_link = os.path.join("models", "minimap_cnn_latest.pth")
    try:
        # Remove existing symlink if it exists
        if os.path.exists(latest_link):
            os.remove(latest_link)
        
        # Create symbolic link
        if os.name == 'nt':  # Windows
            import subprocess
            target = os.path.abspath(model_path)
            link = os.path.abspath(latest_link)
            subprocess.run(['mklink', link, target], shell=True)
            print(f"Created symlink: {latest_link} -> {model_filename}")

    except Exception as e:
        print(f"Warning: Could not create symlink: {e}")
        print("Copying the model file instead...")
        import shutil
        shutil.copy2(model_path, latest_link)
        print(f"Copied model to {latest_link}")

if __name__ == "__main__":
    main()