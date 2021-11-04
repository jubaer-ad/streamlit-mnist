from tensorflow.keras import backend
backend.clear_session()
import streamlit as st
from tensorflow.keras.datasets import mnist

st.title('Customizable Neural Network')

num_neurons = st.sidebar.slider('Number of neurons in hidden layer', 1, 64)
num_epochs = st.sidebar.slider('Number of epochs', 1,10)
activation = st.sidebar.text_input('Activation function')

if st.button('Train the model'):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import *
    from tensorflow.keras.callbacks import ModelCheckpoint

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def preprocess_image(images):
        images = images / 255
        return images
    X_train = preprocess_image(X_train)
    X_test = preprocess_image(X_test)

    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    save_cp = ModelCheckpoint('model', save_best_only=True)
    history_cp = tf.keras.callbacks.CSVLogger('history.csv', separator=',')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[save_cp, history_cp])


if st.button('Evaluate the model'):
    import pandas as pd
    import matplotlib.pyplot as plt
    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'])
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model accuracies vs epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'])
    fig