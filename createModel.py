import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images. Makes it easier to train. Img values are 0-255 so ret value is [0,1]

train_images = (x_train/255) - 0.5
test_images = (x_test/255) - 0.5

train_images = train_images.reshape((-1,784)) #60,000 rows and 784 cols
test_images = test_images.reshape((-1,784)) #10,000 rows and 784 cols

print(train_images.shape)
print(test_images.shape)

# 3 layer model, 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons, with softmax function

model = keras.models.Sequential()
model.add( keras.layers.Dense(128, activation ='relu', input_dim=784))
model.add( keras.layers.Dense(128, activation ='relu'))
model.add( keras.layers.Dense(10, activation='softmax'))

#Compile the model
# Loss function measures how well the model did on training, and hten tries to improve it using optimizer

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics= ['accuracy']
)

model.fit(
    train_images,
    keras.utils.to_categorical(y_train), #ex. 2 it expects [0,1,0,...0] 10 dims
    epochs=7,
    batch_size=32, #number of samples per gradient update
    verbose=1
)

model.evaluate(
    test_images,
    keras.utils.to_categorical(y_test)
)


model.save('model.h5')
predictions = model.predict(test_images[:10])
print(np.argmax(predictions, axis=1))
print(y_test[:10])







