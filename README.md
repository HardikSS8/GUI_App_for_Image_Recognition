# Image_Recognition_GUI_APP
This repository is the GUI application built in Python using the Tkinter library which can used to display the label of the user given image as input in application. 

This application will use the trained neural-network model built by using the Keras and TensorFlow library to predict the class or label of the images.

## Requirements

1. Library: Keras, TensorFlow, TKinter

2. Tools: PyCharm

## Project Description

1. Keras is the built-in framework in Python used to develop the deep learning neural-network with the few lines of programming.

2. TensorFlow is a free and open-source software library for dataflow across a neural-network nodes.

-> This application uses a deep neural-network model developed using the keras framework. Here I have used the CIFAR-10 dataset which consists of 60,000 images (50, 000 for training purpose and 10,000 for test purpose) which can be classified into ten classes.

-> This model can be expressed as multi-class classification model because we are classifying images into more than two classes.

-> This classes are as follows. 
   1. Airplane
   2. Automobile
   3. Bird
   4. Cat
   5. Deer
   6. Dog
   7. Frog
   8. Horse
   9. Ship
   10. Truck

-> This dataset can be download at here. [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) or we can import the dataset using the import command from the keras.datasets.

## Getting Setup

1. Download PyCharm: [https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download)

2. Install the Keras and Tensorflow library by using the following commands in the Terminal.
      
        pip install keras
        
        pip install tensorflow

3. Similarly we can add other libraries like numpy, pathlib using the following commands.
        
        pip install numpy
         
        pip install pathlib
  
4. Clone or download this repository <br/>

   -> Instructions on how to clone/download a GitHub repo: [https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) 
   
5. Open PyCharm and import the project:
      
        File -> Open
   -> Select the project folder - aka the project you just cloned/downloaded from GitHub.

## Specifications of Neural-Network Model 

#### Input Layer  : Three dimesional array (RGB) of 32 * 32 image size. (Total 32 * 32 * 3 nodes will be there in input layer)
#### Output Layer : 10 nodes each representing the particular class (0-9) of multi-class problem.

* In this model, I have added 2 convolutional blocks with the 1 input layer and 1 output layer. This convolutional blocks are made up of convolutional layer, Max pooling layer, Dropout layers. This different layers are added to increase the accuracy of the model.

* Convolutional layer will use the sliding window classifier of given size (3 * 3) to check if the particular shape of required image is present in the whole picture or not.

* After the convolutional blocks, we have to pass it through the Flatten to flatten the complex result.

* And now we can compile the model with some parameters like calculation of error using ("categorical_crossentropy"/ "mean_squared_error"), optimizer function ("adam"), metrics as accuracy. And finally train the model using batch size of 64 images and 45 passes.

* Result of training the model ( Total 45 passes )

![Screenshot 2020-08-08 at 2 26 51 PM](https://user-images.githubusercontent.com/35401920/89708141-3a03c680-d992-11ea-9b7a-4d10ecede945.png)

* We can save the trained model and weights in the file to use them anywhere else.

* Finally we can run the python file for predicting the input image given by user.

## Results:

1. At first GUI will be look like as following.

<img src="https://user-images.githubusercontent.com/35401920/89708424-a384d480-d994-11ea-986c-1a7a3899aa32.png" width="400">

-> Here are some result of trained model with different-different images given by user. We can also see the result in the terminal window of PyCharm.

<img src="https://user-images.githubusercontent.com/35401920/89708335-e85c3b80-d993-11ea-9132-f67bedf56c38.png" width="800">

<br /> 

### other results: 
<br />

<img src="https://user-images.githubusercontent.com/35401920/89708412-79331700-d994-11ea-927b-4fbb9c1eea2c.png" width="600">

<br /> <br />

<img src="https://user-images.githubusercontent.com/35401920/89708450-d9c25400-d994-11ea-8675-b033e70c9e2e.png" width="600">
