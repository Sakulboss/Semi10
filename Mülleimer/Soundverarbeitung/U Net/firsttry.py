import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
#import strategy

def encoder_block(filters, inputs):
  x = tf.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
  s = tf.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
  p = tf.MaxPooling2D(pool_size = (2,2), padding = 'same')(s)
  return s, p #p provides the input to the next encoder block and s provides the context/features to the symmetrically opposte decoder block

def baseline_layer(filters, inputs):
  x = tf.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
  x = tf.Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
  return x

def decoder_block(filters, connections, inputs):
  x = tf.Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(inputs)
  skip_connections = tf.concatenate([x, connections], axis = -1)
  x = tf.Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(skip_connections)
  x = tf.Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(x)
  return x

def unet():
  #Defining the input layer and specifying the shape of the images
  inputs = tf.Input(shape = (224,224,1))

  #defining the encoder
  s1, p1 = encoder_block(64, inputs = inputs)
  s2, p2 = encoder_block(128, inputs = p1)
  s3, p3 = encoder_block(256, inputs = p2)
  s4, p4 = encoder_block(512, inputs = p3)

  #Setting up the baseline
  baseline = baseline_layer(1024, p4)

  #Defining the entire decoder
  d1 = decoder_block(512, s4, baseline)
  d2 = decoder_block(256, s3, d1)
  d3 = decoder_block(128, s2, d2)
  d4 = decoder_block(64, s1, d3)

  #Setting up the output function for binary classification of pixels
  outputs = tf.Conv2D(1, 1, activation = 'sigmoid')(d4)

  #Finalizing the model
  model = tf.Model(inputs = inputs, outputs = outputs, name = 'Unet')

  return model

def load_images(imgsort, masksort, image_dir, mask_dir):
  '''
  Takes the directories and image/mask names and reads them using cv2
  '''

  images, masks = [], []

  for img, msk in tqdm(zip(imgsort, masksort), total=len(imgsort), desc='Loading Images and Masks'):
    image = cv2.imread(image_dir + img, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_dir + msk, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256))

    images.append(image)
    masks.append(mask)

    del image, mask

  return images, masks

#Plotting images for sanity check
def plot_image_with_mask(image_list, mask_list, num_pairs = 4):
    '''
    This functions takes the image and mask lists and prints 4 random pairs of images and masks
    kl
    '''
    plt.figure(figsize = (18,9))
    for i in range(num_pairs):
        idx = random.randint(0, len(image_list))
        img = image_list[idx]
        mask = mask_list[idx]
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f'Real Image, index = {idx}')
        plt.axis('off')
        plt.subplot(2, 4, i + num_pairs + 1)
        plt.imshow(mask)
        plt.title(f'Segmented Image, index = {idx}')
        plt.axis('off')
        del img, mask

# Setting dice coefficient to evaluate our model
def dice_coeff(y_true, y_pred, smooth = 1):
    intersection = K.sum(y_true*y_pred, axis = -1)
    union = K.sum(y_true, axis = -1) + K.sum(y_pred, axis = -1)
    dice_coeff = (2*intersection+smooth) / (union + smooth)
    return dice_coeff

with tf.distribute.Strategy.scope(self=''): #this line allocates multiple GPUs for training in Kaggle
    unet = unet()
    unet.compile(loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy', dice_coeff])

#Defining early stopping to regularize the model and prevent overfitting
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

#Training the model with 50 epochs (it will stop training in between because of early stopping)
unet_history = unet.fit(train_data, validation_data = [val_data],
                        epochs = 50, callbacks = [early_stopping])

#Function to plot the predictions with orginal image, original mask and predicted mask
def plot_preds(idx):
    '''
    This function plots a test image, it's actual mask and the predicted
    mask side by side.
    '''

    plt.figure(figsize = (15, 15))
    test_img = images_test[idx]
    test_img = tf.expand_dims(test_img, axis = 0)
    test_img = tf.expand_dims(test_img, axis = -1)
    pred = unet.predict(test_img)
    pred = pred.squeeze()
    thresh = pred > 0.5
    plt.subplot(1,3,1)
    plt.imshow(images_test[idx])
    plt.title(f'Original Image {idx}')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(masks_test[idx])
    plt.title('Actual Mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(thresh)
    plt.title('Predicted Mask')
    plt.axis('off')

#plotting 10 random images with their true and predicted masks
for i in [random.randint(0, 2000) for i in range(10)]:
    plot_preds(i)

#Defining early stopping to regularize the model and prevent overfitting
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

#Training the model with 50 epochs (it will stop training in between because of early stopping)
unet_history = unet.fit(train_data, validation_data = [val_data],
                        epochs = 50, callbacks = [early_stopping])

plot_image_with_mask(images, masks, num_pairs = 4)

images, masks = load_images(imgsort, masksort, image_dir, mask_dir)