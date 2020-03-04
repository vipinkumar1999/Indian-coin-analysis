
KNOW YOUR COIN!

The main objective of the project is to classify the coin image uploaded by the user to the class it belongs to.

This project is related to numimatics. The dataset was collected from google. Google colab environment is used. 
First procedure is to split the data into Train and Test data. A random split of 0.7 and 0.3 was used.
Next, we train our model using inceptionV3 pre trained model.

About InceptionV3!
Inception-v3 is a convolutional neural network that is trained on more than a million images from the ImageNet database. 
The network is 48 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

The weights of inceptionV3 are assigned to our model(this was done because it was the most appropriate one)
During training the accuracy obtained was 99% and the test accuracy was 84%
An image augmentation was carried out for the purpose simple user experience.
The prediction of food class was done in two steps:
1 predicted
5 predicted

1 PREDICTED:
The top one predicted class is retrieved as the output.

5 PREDICTED:
The top five classes predicted is retrieved as the output.

A confusion matrix is formed, and the confusion matrix is interpreted for future betterment of the model.

CONFUSION MATRIX

A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. 
It allows the visualization of the performance of an algorithm.

True Positive:

Interpretation: You predicted positive and it’s true.

You predicted that a woman is pregnant and she actually is.

True Negative:

Interpretation: You predicted negative and it’s true.

You predicted that a man is not pregnant and he actually is not.

False Positive: (Type 1 Error)

Interpretation: You predicted positive and it’s false.

You predicted that a man is pregnant but he actually is not.

False Negative: (Type 2 Error)

Interpretation: You predicted negative and it’s false.

You predicted that a woman is not pregnant but she actually is.

Just Remember, We describe predicted values.


