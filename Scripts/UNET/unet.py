# %% [markdown]
# # Importing Libraries 

# %%
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
history = History()

# %%
keras.__version__

# %%
tf.__version__

# %%
len(tf.config.list_physical_devices('GPU'))>0

# %% [markdown]
# # Defining Constants

# %%
SEED = 909
# IMG_SIZE (tuple): Target size for the images.
# SEED (int): Random seed for reproducibility.
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

IMAGE_HEIGHT =128
IMAGE_WIDTH =128

IMG_SIZE =(IMAGE_HEIGHT,IMAGE_WIDTH)

data_dir = 'Slices'
# data_dir_image = os.path.join(data_dir,'MRIs')
# data_dir_mask = os.path.join(data_dir,'Masks')
data_dir_train =os.path.join(data_dir,'Train')
data_dir_train_image = os.path.join(data_dir_train,'MRIs')
data_dir_train_mask = os.path.join(data_dir_train,'Mask')

data_dir_test =os.path.join(data_dir,'Test')
data_dir_test_image = os.path.join(data_dir_test,'MRIs')
data_dir_test_mask = os.path.join(data_dir_test,'Mask')

NUM_TRAIN = 2502
NUM_TEST =575

NUM_OF_EPOCHS = 50



# %% [markdown]
# # Image Augmentation

# %% [markdown]
# ### Data Generator
# 

# %%
def create_custom_generator(img_dir, mask_dir, batch_size, img_size, seed):
    data_gen_args = dict(rescale=1./255)
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    img_generator = img_datagen.flow_from_directory(
        img_dir,
        target_size=img_size,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=img_size,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed
    )

    while True:
        img_batch = img_generator.next()
        mask_batch = mask_generator.next()
        yield (img_batch, mask_batch)


# %%
# train_generator=create_segmentation_generator_train(data_dir_train_image,data_dir_train_mask,BATCH_SIZE_TRAIN)
train_generator = create_custom_generator(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN, IMG_SIZE, SEED)
test_generator = create_custom_generator(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST, IMG_SIZE, SEED)


# %% [markdown]
# ### Other Functions

# %%
def display(display_list):
    plt.figure(figsize=(15,15))

    title =['Input Image','True Mask','Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),cmap='gray')
    plt.show()

# %%
def show_dataset(datagen,num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0],mask[0]])

# %% [markdown]
# ### Testing Datagenerator

# %%
show_dataset(train_generator,3)

# %% [markdown]
# ## Metrics Functions

# %%
def iou_score(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# %%
def f1_score(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    precision = intersection / (K.sum(y_pred_f) + smooth)
    recall = intersection / (K.sum(y_true_f) + smooth)
    return 2 * (precision * recall) / (precision + recall + smooth)

# %% [markdown]
# ## Losses

# %%
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5*tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5*dice_loss(y_true, y_pred)

# %% [markdown]
# # Models

# %%
EPOCH_STEP_TRAIN = NUM_TRAIN//BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST//BATCH_SIZE_TEST

# %% [markdown]
# ### UNETv1

# %%
def unet(n_levels,initial_features=32,n_blocks=2,kernel_size=3,pooling_size=2,in_channels=1,out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT,IMAGE_WIDTH,in_channels)) # Input layer
    x = inputs

    convpars = dict(kernel_size=kernel_size,activation='relu',padding='same')

    # downstream path
    skips ={}

    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features*(2**level), **convpars)(x)
        if level < n_levels - 1:
            skips[level]=x
            x=keras.layers.MaxPool2D(pooling_size)(x)
    
    # upstream path
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features*(2**level),strides=pooling_size,**convpars)(x)
        x = keras.layers.Concatenate()([x,skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features*2**level, **convpars)(x)


    #  output

    x = keras.layers.Conv2D(out_channels,kernel_size=1,activation='sigmoid',padding='same')(x)

    return keras.Model(inputs=[inputs],outputs=[x],name=f'UNET-L{n_levels}-F{initial_features}')


model = unet(4)
model.compile(optimizer = 'adam',loss=dice_loss,metrics =['accuracy', iou_score, f1_score])
model.summary()

# %% [markdown]
# ### UNETv2
# 

# %%
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization

# def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
#     inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
#     x = inputs
#     convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
#     skips = {}

#     # Downstream path
#     for level in range(n_levels):
#         for _ in range(n_blocks):
#             x = Conv2D(initial_features * 2**level, **convpars)(x)
#             x = BatchNormalization()(x)
#         if level < n_levels - 1:
#             skips[level] = x
#             x = MaxPooling2D(pooling_size)(x)
#             x = Dropout(0.3)(x)  # Add dropout to avoid overfitting

#     # Bottleneck
#     for _ in range(n_blocks):
#         x = Conv2D(initial_features * 2**n_levels, **convpars)(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.3)(x)

#     # Upstream path
#     for level in reversed(range(n_levels - 1)):
#         x = Conv2DTranspose(initial_features * 2**level, strides=pooling_size, **convpars)(x)
#         x = Concatenate()([x, skips[level]])
#         for _ in range(n_blocks):
#             x = Conv2D(initial_features * 2**level, **convpars)(x)
#             x = BatchNormalization()(x)
#             x = Dropout(0.3)(x)

#     # Output layer
#     x = Conv2D(out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)

#     return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')

# # Compile the model
# model = unet(4)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print model summary
# model.summary()


# %% [markdown]
# ### UNETv3

# %%
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization

# def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
#     inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
#     x = inputs
#     convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
#     skips = {}

#     # Downstream path
#     for level in range(n_levels):
#         for _ in range(n_blocks):
#             x = Conv2D(initial_features * 2**level, **convpars)(x)
#             x = BatchNormalization()(x)
#         if level < n_levels - 1:
#             skips[level] = x
#             x = MaxPooling2D(pooling_size)(x)
#             x = Dropout(0.3)(x)  # Add dropout to avoid overfitting

#     # Bottleneck
#     for _ in range(n_blocks):
#         x = Conv2D(initial_features * 2**n_levels, **convpars)(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.3)(x)

#     # Upstream path
#     for level in reversed(range(n_levels - 1)):
#         x = Conv2DTranspose(initial_features * 2**level, strides=pooling_size, **convpars)(x)
#         x = Concatenate()([x, skips[level]])
#         for _ in range(n_blocks):
#             x = Conv2D(initial_features * 2**level, **convpars)(x)
#             x = BatchNormalization()(x)
#             x = Dropout(0.3)(x)

#     # Output layer
#     x = Conv2D(out_channels, kernel_size=1, activation='sigmoid', padding='same')(x)

#     return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')

# # Compile the model
# model = unet(4)
# model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', iou_score, f1_score])

# # Print model summary
# model.summary()


# %% [markdown]
# ### Training

# %%
trained_model = model.fit(x=train_generator,steps_per_epoch=EPOCH_STEP_TRAIN,validation_data=test_generator,validation_steps=EPOCH_STEP_TEST,epochs=NUM_OF_EPOCHS, callbacks=history)

# %%
print(trained_model.history.items)

# %% [markdown]
# ## Training plots

# %%
import matplotlib.pyplot as plt
plt.plot(trained_model.history['loss'])

# %% [markdown]
# ### Saving model

# %%
model.save("UNET_nd_dl.h5")

# %% [markdown]
# ### Visualizing Predictions

# %%
test_generator = create_custom_generator(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST, IMG_SIZE, 1)

# %%
def show_prediction(datagen,num=1):
    for i in range(num):
        image,mask=next(datagen)
        pred_mask=model.predict(image)[0]
        pred_mask = pred_mask > 0.5
        display([image[0],mask[0],pred_mask])

# %%
show_prediction(test_generator,15)


