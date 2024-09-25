import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the model architecture
class AutonomousDrivingModel(nn.Module):
    def __init__(self):
        super(AutonomousDrivingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        self.fc1 = nn.Linear(64 * 1 * 18, 1164)  # Adjust based on image size
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # Output steering angle
        return x

# Custom dataset class to load images and labels
class SteeringDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, sep=' ', names=['filename', 'angle'], index_col=False)
        self.data['angle'] = self.data['angle'].apply(lambda x: x.split(',')[0]).astype(float) * 3.14159265 / 180
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        angle = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(angle).float()

# Define transforms for the dataset
image_size = (66, 200)  # Resize images
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Load dataset
train_split = 0.8
batch_size = 32
csv_file = "/home/rakshithram/SDC_project/data.txt"
root_dir = "/home/rakshithram/SDC_project/data"

dataset = SteeringDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for training
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutonomousDrivingModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training and validation loop
num_epochs = 30
train_losses = []
val_losses = []

print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, angles) in enumerate(train_loader):
        images, angles = images.to(device), angles.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, angles) in enumerate(val_loader):
            images, angles = images.to(device), angles.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), angles)
            val_loss += loss.item()
            
            #if batch_idx % 10 == 0:  # Print every 10 batches
                #print(f"Validation - Batch [{batch_idx}/{len(val_loader)}], Loss: {loss.item():.4f}")
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(" ")
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(" ")

# Save model
model_save_path = 'save/final_model.pth'
if not os.path.exists('save'):
    os.makedirs('save')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

print("Training complete.")
