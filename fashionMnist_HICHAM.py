from __future__ import print_function

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
import sys


batch_size = 120
num_classes = 10
epochs = 15

filepath = "C:/Users/hicha/Desktop/CNN-HICHAM/model_fashion_mnist_hicham.h5"

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(2, 2), activation = 'relu', input_shape = input_shape))
fashion_model.add(Conv2D(64, (2, 2), activation = 'relu'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2)))
fashion_model.add(Dropout(0.15))






fashion_model.add(Flatten())
fashion_model.add(Dense(64, activation='relu'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])
model.summary()

#fashion_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

#stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

check = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

tensor_b  =keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,patience=3, min_lr=0.0001)


model.fit(x_train, y_train,
		callbacks=[check ,reduce_lr , tensor_b],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# fits the model on batches with real-time data augmentation:
gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
		
generator = gen.flow(x_train, y_train, batch_size) 


#fashion_model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                   # steps_per_epoch=len(x_train) / 32, epochs=epochs,callbacks=[reduce_lr , tensor_b])


#fashion_model.fit(x_train, y_train, epochs=epochs,callbacks=[reduce_lr , tensor_b])
score = fashion_model.evaluate(x_test, y_test, verbose = 0)





print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model
model.save("C:/Users/hicha/Desktop/CNN-HICHAM/model_fashion-HICHAM" + str(epochs) + ".h5")


#si l'accurcy devient stable alors il faut diminuer le learning rate ( ex : coef = 0.5 -> div par deux )
