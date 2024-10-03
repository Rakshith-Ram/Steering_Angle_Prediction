# Steering Angle Prediction

Click here for NVIDIAs article: [Link](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

Access the paper: [Link](https://arxiv.org/pdf/1604.07316v1)

This is a PyTorch implementation of NVIDIAs model with some minor modifications. 


### Gif of the output:
![](https://github.com/Rakshith-Ram/Steering_Angle_Prediction/blob/main/demo_on_dataset.gif)

### Download the dataset from here: 
[Link](https://drive.google.com/file/d/1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B/view?pli=1)

Approximately 63,000 images, 3.1GB. Data was recorded around Rancho Palos Verdes and San Pedro California.

Data format is as follows: filename.jpg angle,year-mm-dd hr:min:sec:millisec


Use python3 "train.py" to train the model

Use python3 "run.py" to run the model on a live webcam feed

Use python3 "run_dataset.py" to run the model on the dataset
