import torch
import torch.nn as nn
import cv2  # For webcam access
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_model import AutonomousDrivingModel

# Define transforms (same as during training)
image_size = (66, 200)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Initialize model and load the saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutonomousDrivingModel().to(device)
model.load_state_dict(torch.load('/home/rakshithram/SDC_project/final_model.pth', map_location=device))
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

# Setup figure windows for visualization
fig1, ax1 = plt.subplots(figsize=(3, 3))  # Steering wheel (predicted angle) window
fig2, ax2 = plt.subplots(figsize=(6, 3))  # Image window

# Add titles to the windows
fig1.suptitle("Predicted Steering Angle", fontsize=12)

# Initialize video capture with the MP4 file
video_file_path = "/home/rakshithram/SDC_project/sample.mp4"  # Change this to the path of your video file
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_file_path}.")
    print("\n Trying to detect webcam!")
    # Initialize the webcam (camera 0 by default)
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop over the frames from the webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break
    
    # Convert the frame (BGR from OpenCV) to RGB and then to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply the same transforms as during training
    transformed_image = transform(pil_image).unsqueeze(0).to(device)

    # Predict steering angle and use .detach() before calling numpy()
    predicted_angle = model(transformed_image).detach().cpu().numpy()[0][0]
    predicted_angle_deg = predicted_angle * 180 / 3.14159265

    # Display the input image in a separate window (fig2, ax2)
    ax2.clear()
    ax2.imshow(pil_image)
    ax2.set_title(f"Predicted: {predicted_angle_deg:.2f}°")
    ax2.axis('off')

    # Visualize the steering wheel image rotation for predicted angle (fig1, ax1)
    draw_steering_wheel_image(predicted_angle_deg, ax1, steering_wheel_image_path)

    # Update all figures
    fig1.canvas.draw()
    fig2.canvas.draw()

    # Display the frame using OpenCV
    #cv2.imshow('Webcam Feed', frame)

    print(f"Predicted steering angle: {predicted_angle_deg:.2f}°")

    # Pause to create a video effect and handle key presses
    plt.pause(0.001)  # Adjust the pause duration to control playback speed
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
