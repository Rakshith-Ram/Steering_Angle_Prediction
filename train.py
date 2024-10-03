import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")  # This will suppress all warnings

from pytorch_model import AutonomousDrivingModel  # Importing the prebuilt model


# Custom dataset class to load images and labels
class SteeringDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Load CSV data
        self.data = pd.read_csv(csv_file, sep=' ', names=['filename', 'angle'], index_col=False)
        # Convert angles from degrees to radians
        self.data['angle'] = self.data['angle'].apply(lambda x: float(x.split(',')[0]) * 3.14159265 / 180)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        angle = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(angle).float()

# Define transforms
image_size = (66, 200)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Load dataset
csv_file = "/home/rakshithram/Steering_Angle_Prediction/data.txt"
root_dir = "/home/rakshithram/Steering_Angle_Prediction/data"
dataset = SteeringDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# Split dataset into training and validation sets
train_split = 0.8
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutonomousDrivingModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # learning rate

# Training loop
num_epochs = 30
train_losses = []
val_losses = []

print("\nStarting training...")

print(f"\n   for {num_epochs} epochs,  batch size {batch_size},  training data = {100*train_split} %,  validation data = {100-(100*train_split)} %")

for epoch in range(num_epochs):
    print(f"\nEPOCH ({epoch + 1}/{num_epochs}) -------------------------------------------------------------\n")
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_idx, (images, angles) in enumerate(train_loader):
        images, angles = images.to(device), angles.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs.squeeze(), angles)
        
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()

        # Print progress
        if (batch_idx+1)%100 == 0:
            print(f"   Batch ({batch_idx + 1}/{len(train_loader)}), Loss: {loss.item():.4f}")
    
    # Calculate average training loss for the epoch
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, angles) in enumerate(val_loader):
            images, angles = images.to(device), angles.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), angles)
            val_loss += loss.item()

            # Print validation batch progress
            if (batch_idx+1)%100 == 0:
                print(f"   Validation Batch ({batch_idx + 1}/{len(val_loader)}), Loss: {loss.item():.4f}")
    
    # Calculate average validation loss for the epoch
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Print summary of the epoch
    print(f"\nAfter Epoch ({epoch + 1}) --> Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save model
save_dir = "/home/rakshithram/Steering_Angle_Prediction/save/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_save_path = os.path.join(save_dir, 'final_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}")

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\n ---------------------------------------- Training complete. ----------------------------------------")
