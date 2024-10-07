# Steering Angle Prediction

Click here for NVIDIAs article: [Link](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

Access the paper: [Link](https://arxiv.org/pdf/1604.07316v1)

This is a PyTorch implementation of NVIDIAs model with some minor modifications. 


### Gif of the output:
![](https://github.com/Rakshith-Ram/Steering_Angle_Prediction/blob/main/demo_on_dataset.gif)

### Download the dataset from here: 
[Link](https://drive.google.com/file/d/1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B/view?pli=1)

Approximately 63,000 images, 3.1GB. Data was recorded around Rancho Palos Verdes and San Pedro California.

Data format is as follows: _filename.jpg angle,year-mm-dd hr:min:sec:millisec_

### Instructions:

The [model.py](https://github.com/Rakshith-Ram/Steering_Angle_Prediction/blob/main/model.py)  file contains the PyTorch implementation of the architecture.

Use [train.py](https://github.com/Rakshith-Ram/Steering_Angle_Prediction/blob/main/train.py)  to train the model using the above architecture on the dataset.

Use [run.py](https://github.com/Rakshith-Ram/Steering_Angle_Prediction/blob/main/run.py)  to run the model on a recorded video or a live webcam feed.

Use [visualize.py](https://github.com/Rakshith-Ram/Steering_Angle_Prediction/blob/main/visualize.py)  to run the model on the dataset and get the visualization of the steering angle.

The `model.py` file contains the PyTorch implementation of the architecture.


### Credits
- Credits: https://github.com/SullyChen/Autopilot-TensorFlow
