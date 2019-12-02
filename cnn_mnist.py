import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import LambdaCallback
from utils import *

class Event:
    def __init__(self, key):
        self.key = key
    pass

## Load dataset from build in MNIST data loader
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## Check the data
def close_on_key(event):
    plt.close()

fig = plt.figure(figsize=(12,6))
fig.canvas.mpl_connect('key_press_event', close_on_key)
ax = fig.subplots(3,6)
ax = np.reshape(ax, ax.size)
print(y_train[:ax.size])

for i, a in enumerate(ax):
    a.set_axis_off()
    a.imshow(np.squeeze(x_train[i]), cmap='Greys')

print("Close the window to continue!")
plt.show()

## Reshaping the array similar to my from scratch impl.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#Crop out a validation set
x_train, x_val = np.split(x_train, (55000,))
y_train, y_val = np.split(y_train, (55000,))

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_val', x_val.shape[0])
print('Number of images in x_test', x_test.shape[0])

## Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(14, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(14, kernel_size=(3,3), input_shape=(13, 13, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()

## Create an evaluation callback for integrating the test routine
def run_eval(epoch, logs):
    eval = model.evaluate(x_test, y_test)
    logs['test_loss'] = eval[0]
    logs['test_acc'] = eval[1]

## Convert to tensorflow graph
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
    )

## Run training session
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=2,
    validation_data=(x_val, y_val),
    callbacks=[
        LambdaCallback(
            on_epoch_end = run_eval
            )
        ]
    )

## Plot the Losses
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', close_on_key)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.plot(history.history['test_loss'], label='test')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
print("Close the window to continue!")
plt.show()

## Plot the Accuracy
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', close_on_key)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='val')
plt.plot(history.history['test_acc'], label='test')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
print("Close the window to continue!")
plt.show()

## Plot Layer Weights
fig, ax = plt.subplots(2,2)
ax = ax.flatten()
titles = ["Conv2D_1", "Conv2D_2", "Dense_1", "Dense_2"]
ids = [0, 2, 5, 7]
fig.canvas.mpl_connect('key_press_event', close_on_key)
for i, a in enumerate(ax):
    weights, biases = model.layers[ids[i]].get_weights()
    a.set_title(titles[i])
    a.hist(weights.flatten())
    a.set_xlabel("Value Bins")
    a.set_ylabel("Occurrence")
plt.tight_layout()
plt.show()

## Qualitative examples:
def sample_handler(event):
    if event.key is 'escape':
        plt.close()
    elif event.key is 'enter':
        sample = np.random.randint(x_test.shape[0])
        gt = y_test[sample]
        pred = model.predict(x_test[sample].reshape(1, 28, 28, 1)).argmax()
        plt.imshow(x_test[sample].reshape(28, 28),cmap='Greys')
        plt.title("Prediction: {}, Label: {}".format(pred, gt))
        print("Prediction:", pred, "Label:", gt)
        plt.draw()


fig = plt.figure(figsize=(6,4))
fig.canvas.mpl_connect('key_press_event', sample_handler)
sample_handler(Event('enter'))
plt.show()

## Find Miss-Classifications