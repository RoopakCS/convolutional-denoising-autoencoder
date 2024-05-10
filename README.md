# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Convolution Autoencoder Network Model

## DESIGN STEPS

### STEP 1:
Download and split the dataset into training and testing datasets

### STEP 2:
rescale the data as that the training is made easy

### STEP 3:
create the model for the program , in this experiment we create to networks , one for encoding and one for decoding Write your own steps

## PROGRAM

```python
Name : Roopak C S
Reg.No : 212223220088
```
### Importing modules
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```
### Importing Dataset
```python
(x_train, _), (x_test, _) = mnist.load_data()
```
```python
x_train.shape
```
```python
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
### Creating the model
```python
input_img = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x = layers.MaxPooling2D((2,2),padding='same')(x)
x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
```
```python
autoencoder = keras.Model(input_img, decoded)
```
```python
autoencoder.summary()
```
### Compiling and fitting the model
```python
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,epochs=2,batch_size=128,shuffle=True,validation_data=(x_test_noisy, x_test_scaled))
```
```python
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/RoopakCS/convolutional-denoising-autoencoder/assets/139228922/c932f7ee-ed6b-476c-9150-2f4e52a29c21)


### Model Summary:
![image](https://github.com/RoopakCS/convolutional-denoising-autoencoder/assets/139228922/0e189668-5805-4d7a-8dbe-355b4564aa87)


### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/RoopakCS/convolutional-denoising-autoencoder/assets/139228922/6a51dce8-e883-4f52-979a-b6a34c5aa1a4)

## RESULT:
Thus we have successfully developed a convolutional autoencoder for image denoising application.
