import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

# Define the model architecture (same as before)
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
class SteeringDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, sep=' ', names=['filename', 'angle'], usecols=[0, 1], index_col=False)
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

# Define transforms (same as during training)
image_size = (66, 200)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Load the dataset
csv_file = "/home/rakshithram/SDC_project/data.txt"
root_dir = "/home/rakshithram/SDC_project/data"
dataset = SteeringDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize model and load the saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutonomousDrivingModel().to(device)
model.load_state_dict(torch.load('save/final_model.pth', map_location=device))
model.eval()

# Function to draw and rotate steering wheel image
def draw_steering_wheel_image(angle_deg, ax, steering_wheel_img):
    ax.clear()  # Clear the previous image
    
    # Load the steering wheel image (if not already loaded)
    steering_wheel_img = Image.open(steering_wheel_img)
    
    # Apply the rotation to the image using the angle
    rotated_img = steering_wheel_img.rotate(-angle_deg)  # Negative to rotate correctly
    
    # Display the rotated image
    ax.imshow(rotated_img)
    
    # Remove axis for clean visualization
    ax.axis('off')

# Path to the steering wheel image
steering_wheel_image_path = "/home/rakshithram/SDC_project/steering_wheel.png"

# Predict steering angle for each image and visualize in separate windows
fig1, ax1 = plt.subplots(figsize=(3, 3))  # Steering wheel (predicted angle) window
fig2, ax2 = plt.subplots(figsize=(6, 3))  # Image window
fig3, ax3 = plt.subplots(figsize=(3, 3))  # Steering wheel (actual angle) window

# Add titles to the windows
fig1.suptitle("Predicted Steering Angle", fontsize=12)
fig2.suptitle("Input Image", fontsize=12)
fig3.suptitle("Actual Steering Angle", fontsize=12)

for images, true_angle in data_loader:
    images = images.to(device)
    
    # Predict steering angle and use .detach() before calling numpy()
    predicted_angle = model(images).detach().cpu().numpy()[0][0]
    predicted_angle_deg = predicted_angle * 180 / 3.14159265
    true_angle_deg = true_angle.cpu().numpy()[0] * 180 / 3.14159265
    
    # Display the input image in a separate window (fig2, ax2)
    ax2.clear()
    ax2.imshow(transforms.ToPILImage()(images.cpu().squeeze()))
    ax2.set_title(f"Actual: {true_angle_deg:.2f}° | Predicted: {predicted_angle_deg:.2f}°")
    ax2.axis('off')
    
    # Visualize the steering wheel image rotation for predicted angle (fig1, ax1)
    draw_steering_wheel_image(predicted_angle_deg, ax1, steering_wheel_image_path)
    
    # Visualize the steering wheel image rotation for actual angle (fig3, ax3)
    draw_steering_wheel_image(true_angle_deg, ax3, steering_wheel_image_path)
    
    # Update all figures
    fig1.canvas.draw()
    fig2.canvas.draw()
    fig3.canvas.draw()
    
    # Pause to create a video effect
    plt.pause(0.0033)  # Adjust the pause duration to control playback speed

plt.show()
