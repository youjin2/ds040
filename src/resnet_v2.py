import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Conv2D,
    BatchNormalization,
    AveragePooling2D,
    Flatten,
    add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import disable_eager_execution

from utils import config_gpu
_ = config_gpu()
disable_eager_execution()


def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)

    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs

    # for resnet v1
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    # for resnet v2
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    return x


def resnet_v2(input_shape, depth, num_classes=10):

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (e.g. 20, 101, 164, ...)')

    num_filters_in = 16
    num_res_blocks = int((depth-2)/9)

    inputs = Input(shape=input_shape)

    # input conv
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # downsample feature map 1/2 times for every 2n iterations
    # expand output filter size 2 times for every 2n iterations
    for stage in range(3):

        # residual block
        for res_block in range(num_res_blocks):
            strides = 1
            activation = 'relu'
            batch_normalization = True

            # downsample the feature map with 1/2 for every 2n iterations
            if stage == 0:
                num_filters_out = num_filters_in*4
                # if res_block == 0:
                #     activation = None
                #     batch_normalization = False
            else:
                num_filters_out = num_filters_in*2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             strides=1,
                             activation='relu',
                             conv_first=False)

            # downsampling the input x is required for skip-connection
            # note that activation and BN are not applied! (only resizing the feature map)
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            # shortcut-connection
            x = add([x, y])

        # expand filter size for every 2n iterations
        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8, padding='same')(x)
    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# train configurations
batch_size = 128
epochs = 200
data_augmentation = True


# resnet configuration
n = 18
depth = n*9+2


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize data.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# resacle to [-1, 1]
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

# to categorical
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

input_size = x_train.shape[1:]


# build model
model = resnet_v2(input_size, depth)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])


# save path
save_dir = '../models/cifar10_resnet_v2_{}'.format(int(depth))
os.makedirs(save_dir, exist_ok=True)
plot_model(model, to_file='../figures/cifar10_resnet_v2_{}.png'.
           format(int(depth)), show_shapes=True)


# prepare callbacks for model saving and for learning rate adjustment.
file_path = save_dir
checkpoint = ModelCheckpoint(filepath=file_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]


# data generator
# this will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False
)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)


# train model
# fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1,
                    steps_per_epoch=len(x_train)//batch_size,
                    callbacks=callbacks)
