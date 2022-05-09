import os
from pathlib import Path
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.utils.vis_utils import plot_model


def build_model(input_shape=(128, 256, 1)):
    regularizer = tf.keras.regularizers.l2(0.001)
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv1', activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPool2D((9, 9), strides=(3, 3)))

    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv2', activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPool2D((7, 7), strides=(3, 3)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv3', activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPool2D((5, 5), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv4', activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPool2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='fc2'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


data_dir = Path(r'dataset')
df_train_images = pd.DataFrame({'path': list(data_dir.glob(r'wave/training/augm_*/*.png'))})
df_train_images['label'] = df_train_images['path'].map(lambda x: x.parent.stem[5:])
df_train_images['path'] = df_train_images['path'].astype(str)
df_train_images = df_train_images.sample(frac=1)

df_test_images = pd.DataFrame({'path': list(data_dir.glob(r'wave/testing/*/*.png'))})
df_test_images['label'] = df_test_images['path'].map(lambda x: x.parent.stem)
df_test_images['path'] = df_test_images['path'].astype(str)
df_test_images = df_test_images.sample(frac=1)

batch_size = 128
img_height = 128
img_width = 256
val_split = 0.4
num_train_images = len(df_train_images.index)
num_val_images = int(num_train_images * val_split)
print(num_train_images, num_val_images)

data_generator = ImageDataGenerator(validation_split=val_split)

train_generator = data_generator.flow_from_dataframe(df_train_images,
                                                     x_col='path',
                                                     y_col='label',
                                                     subset="training",
                                                     color_mode="grayscale",
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     seed=42)

val_generator = data_generator.flow_from_dataframe(df_train_images,
                                                   x_col='path',
                                                   y_col='label',
                                                   subset="validation",
                                                   color_mode="grayscale",
                                                   target_size=(img_height, img_width),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=42)

class_names = [key for key in train_generator.class_indices.keys()]
print(class_names)

model = build_model()
model.summary()

plot_model(model, to_file='wave/model_wave_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB')

tensor_board = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
log_file_path = 'logs/wave_training_best.log'
csv_logger = CSVLogger(log_file_path, append=False)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',
    min_delta=0.001, cooldown=0, min_lr=0)
trained_models_path = 'models/best_model_wave.hdf5'
model_checkpoint = ModelCheckpoint(trained_models_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                               save_weights_only=False)
callbacks = [tensor_board, model_checkpoint, csv_logger, reduce_lr]

model.fit(train_generator,
          batch_size=batch_size,
          epochs=150,
          validation_data=val_generator,
          callbacks=callbacks,
          verbose=1)

