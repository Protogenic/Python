import numpy as np
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist                                         #Import built-in dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #Separation dataset on train and test images and labels

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),       #Input layer 28x28 pixels
    keras.layers.Dense(128, activation = tf.nn.relu), #128 hidden layers, relu return x if x>0, else return 0
    keras.layers.Dense(10, activation=tf.nn.softmax)  #10 output layers, each one match one of types of images, softmax return max value
])

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy') #Compiles model with optimizer and loss atribute
model.fit(train_images, train_labels, epochs=5)                           #5 epochs of learning on train datasets
model.evaluate(test_images, test_labels)                                  #Checks acc and loss on test images
predictions = model.predict(test_images)                                  #Predictions 

print(predictions[0]) #Output array with predictions 
print(test_labels[0]) #Output class of test image
