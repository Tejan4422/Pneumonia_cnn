# Pneumonia_cnn
Convolutional Neural Network to detect whether a person's MRI scan has Pneumonia or not
# Overview
  * Building a CNN with TensorFlow backend
  * Test on Single Image
  * Model Production with Flask
* Packages Used : 
  * pandas, numpy , keras, tensorFlow, seaborn, flask, matplotlib
* CNN Structure : 
    1. CONV2D with filter size 3*3, No Padding, No Strides, activation Relu, Convert i/p to 64*64
    2. MaxPooling with Pool size 2*2
    3. CONV2D with filter size 3*3, No Padding, No Strides, activation Relu
    4. MaxPooling with Pool size 2*2
    5. Full Connection Activation Sigmoid, Optimizer = Adam, binary Cross Entropy
    6. Model Accuracy on training Data 94%
    7. Model Accuracy on Testing 88%
    ![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/model_summary.png "Model Summary")

* Case Studies : 
Following is the MRI of a Normal Person Pneumonia -ve
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/normal.png "Negative Case")
Following is the MRI of a Pneumonia +ve
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/pneumonia.png "Positive Case")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/model_summary.png "Model Summary")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/traning_val_accuracy.png "Model Summary")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/loss.png "Model Summary")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/Model_deployment.png "Final Product")



