import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import models
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, LambdaCallback
from keras import regularizers





# Hyper parameters
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
model_name = 'cnn_cifar10.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

## Check the data
def close_on_key(event):
    plt.close()

fig = plt.figure(figsize=(12,6))
fig.canvas.mpl_connect('key_press_event', close_on_key)
ax = fig.subplots(3,6)
ax = np.reshape(ax, ax.size)
fig.suptitle("Labels: " + str(np.squeeze(y_train[:ax.size])))

for i, a in enumerate(ax):
    a.set_axis_off()
    a.imshow(np.squeeze(x_train[i]), cmap='Greys')

print("Close the window to continue!")
#plt.tight_layout()
plt.show()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# 24bit colors to 32bit float colors
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_tensor = (1, x_train.shape[1], x_train.shape[2], x_train.shape[3])



# Load or Configure model
load_model = input("Load a pretrained model? (filename or keep blank):")
if load_model:
    # Load the model
    model = models.load_model(load_model)
    model.summary()
else:
    # Configure the model
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3),
        padding='same',
        input_shape=x_train.shape[1:]
        ))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(Activation('softmax'))
    
    # Compile the model with Adam optimizer
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.summary()

    def run_eval(epoch, logs):
        print("Test:")
        eval = model.evaluate(x_test, y_test)
        logs['test_loss'] = eval[0]
        logs['test_acc'] = eval[1]
        print('Test loss:', eval[0], 'Test accuracy:', eval[1])

    # Train the model
    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        shuffle=True,
        callbacks=[
            EarlyStopping(),
            LambdaCallback(
                on_epoch_end = run_eval
                )
            ]
        )
    # Save model and weights
    model.save(model_name)
    
    ## Plot the Losses
    fig, ax_l = plt.subplots(1)
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    ax_l.plot(history.history['loss'], label='train_loss')
    ax_l.plot(history.history['val_loss'], label='val_loss')
    ax_l.plot(history.history['test_loss'], label='test_loss')
    ax_l.set_ylabel("Loss")
    ax_l.legend(loc=2)

    ## Plot the Accuracy
    ax_r = ax_l.twinx()
    ax_r.plot(history.history['acc'], '--', label='train_acc')
    ax_r.plot(history.history['val_acc'], '--', label='val_acc')
    ax_r.plot(history.history['test_acc'], '--', label='test_acc')
    ax_r.set_ylabel("Accuracy")
    ax_r.legend()

    plt.title("Learning Curve / Training Accuracy")
    plt.xlabel("Epoch")

    print("Close the window to continue!")
    plt.show()


## Plot Layer Weights
fig, ax = plt.subplots(2,2)
ax = ax.flatten()
titles = ["Conv2D_1", "Conv2D_2", "Conv2D_3", "Conv2D_4"]
ids = [0, 2, 6, 8]
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
class Event:
    def __init__(self, key):
        self.key = key
    pass

def sample_handler(event):
    if event.key is 'escape':
        plt.close()
    elif event.key is 'enter':
        sample = np.random.randint(x_test.shape[0])
        gt = y_test[sample].argmax()
        pred = model.predict(x_test[sample].reshape(x_tensor)).argmax()
        plt.imshow(x_test[sample].reshape(x_tensor[1:]))
        plt.title("Prediction: {}, Label: {}".format(labels[pred], labels[gt]))
        print("Prediction:", labels[pred], "Label:", labels[gt])
        plt.draw()


fig = plt.figure(figsize=(6,4))
fig.canvas.mpl_connect('key_press_event', sample_handler)
sample_handler(Event('enter'))

print("-----------------Controls-----------------")
print("Press 'Enter' for next layer.")
print("Press 'Escape' to close the plot window.")
print("Close the window to continue!")
plt.show()



## Find and Viz Mismatch
mispred = model.predict(x_test.reshape(-1, x_tensor[1], x_tensor[2], x_tensor[3])).argmax(axis=1)
mismatch =  mispred != y_test.argmax(axis=1)
print("Num of Mismatches: ", mismatch.sum())
mismatch = np.where(mismatch)[0]
print(mismatch)
mmiter = iter(mismatch)

def mismatch_handler(event):
    if event.key is 'escape':
        plt.close()
    elif event.key is 'enter':
        try:
            sample = next(mmiter)
            print(sample)
        except:
            print("No more samples!")
            plt.close()
            return
        gt = y_test[sample].argmax()
        plt.imshow(x_test[sample].reshape(x_tensor[1:]),cmap='Greys')
        plt.title("Mis-Prediction: {}, Label: {}".format(labels[mispred[sample]], labels[gt]))
        print("Mis-Prediction:", labels[mispred[sample]], "Label:", labels[gt])
        plt.draw()


fig = plt.figure(figsize=(6,4))
fig.canvas.mpl_connect('key_press_event', mismatch_handler)
mismatch_handler(Event('enter'))

print("-----------------Controls-----------------")
print("Press 'Enter' for next layer.")
print("Press 'Escape' to close the plot window.")
print("Close the window to continue!")
plt.show()




## Viz the Activation Potentials
layer_names = [model.layers[id].name for id in ids]
print(layer_names)
layer_iter = iter(layer_names)
sample = 0
name = None

def act_pot_handler(event):
    global name
    global sample
    if event.key is 'escape':
        plt.close()
        return
    elif event.key is 'enter':
        try:
            name = next(layer_iter)
        except:
            print("No more layers!")
            plt.close()
            return
    elif event.key is 'n':
        sample += 1
        
    layer_output = model.get_layer(name).output
    act_model = models.Model(inputs=model.input, outputs=layer_output)
    act = act_model.predict(x_test[sample].reshape(x_tensor))
    _, V, U, C = act.shape
    cells = np.ceil(np.sqrt(C)).astype(int)
    fmap = np.zeros((V*cells,U*cells))
    imn = 0
    for v in range(cells):
        for u in range(cells):
            if imn >= act.shape[-1]:
                break
            vc, uc = v*V, u*U
            fmap[vc:vc+V, uc:uc+U] = act[:,:,:,imn]
            imn += 1
    ax.set_title(name)
    ax.imshow(fmap)
    plt.draw()
    
fig, ax = plt.subplots(1)
fig.canvas.mpl_connect('key_press_event', act_pot_handler)
act_pot_handler(Event('enter'))

print("-----------------Controls-----------------")
print("Press 'Enter' for next layer.")
print("Press 'N' for random sample.")
print("Press 'Escape' to close the plot window.")
print("Close the window to continue!")
plt.show()