# Overfitting in Student Performance Prediction
## Problem Statement
In this project, a neural network is trained using student data.
The model performs very well on the training students.
However, its performance decreases when it is tested on new students.
This project analyzes whether this difference is caused by overfitting.
## Objective
The main objectives of this project are:
1.To build a simple neural network.
2.To train the model using student data.
3.To test the model using unseen student data.
4.To compare training and test accuracy.
5.To identify whether the model is overfitting.
## Concept Used
Overfitting happens when a neural network learns the training data too closely.
Because of this, the model gives very good results on training data but gives poorer
results on new data.
In this project, I identify overfitting by comparing the training accuracy with the
accuracy obtained on unseen student data.
## Dataset
Study hours
Attendance
Assignment completion
Previous score
The dataset contains information about students.
The input features are:
- Study hours
- Attendance percentage
- Assignment completion percentage
- Previous score
The output is the student's result, which is classified as Pass or Fail.
## Methodology
1.I created a student dataset.
2.I divided the data into training data and unseen test data.
3.I normalized the input values.
4.I created a neural network with hidden layers.
5.I trained the network using the training data.
6.I calculated the training accuracy.
7.I tested the trained model using unseen students.
8.I calculated the test accuracy.
9.I compared the training and test accuracy.
10.I analyzed whether the difference indicates overfitting.
## Results
The model performance will be compared using training accuracy and test accuracy.
Training Accuracy:
Test Accuracy:
The difference between the two values will be used to analyze overfitting.
## Conclusion
The model performed better on the training data than on unseen student data.
The difference indicates that the model may be overfitting the training data.
