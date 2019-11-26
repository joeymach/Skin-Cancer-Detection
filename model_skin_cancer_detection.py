#----------------------------IMPORT KAGGLE datasets------------------------
!pip install kaggle
!mkdir .kaggle 
import json
token = {"username":"your-username","key":"your-key"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)
    
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}
!chmod 600 /root/.kaggle/kaggle.json

#------------Downloading and unzipping-------------
!kaggle datasets download kmader/skin-cancer-mnist-ham10000 -p /content/

!apt install unzip


!unzip /content/skin-cancer-mnist-ham10000.zip -d /content/
# Unzip the whole zipfile into /content/data
#!unzip -o data/skin-cancer-mnist-ham10000.zip -d data
# Quietly unzip the image files
!unzip /content/HAM10000_images_part_1.zip -d /content/ham10000_images_part_1/ 
!unzip /content/HAM10000_images_part_2.zip -d /content/ham10000_images_part_2/
# Tell me how many files I unzipped///

#---------------make directories for data---------------------
import os 
import errno
base_dir = 'base_dir'
!mkdir base_dir
image_class = ['nv','mel','bkl','bcc','akiec','vasc','df']

# Make 3 directoris, base_dir, train_dir inside base_dir, and val_dir inside base_dir
# returns error if you rerun the code while making directories, which is why I used try and except
try:
    os.mkdir(base_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

train_dir = os.path.join(base_dir, 'train_dir')
try:
  os.mkdir(train_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

val_dir = os.path.join(base_dir, 'val_dir')
try: 
  os.mkdir(val_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try: 
  for x in image_class:
    os.mkdir(train_dir+'/'+x)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try: 
  for x in image_class:
    os.mkdir(val_dir+'/'+x)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

#--------------splitting data/transfering data------------

import pandas as pd
import shutil
df = pd.read_csv('/content/HAM10000_metadata.csv')


# Set y as the labels
y = df['dx']

#split data
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

# Transfer the images into folders, Set the image id as the index
image_index = df.set_index('image_id', inplace=True)


# Get a list of images in each of the two folders
folder_1 = os.listdir('/content/ham10000_images_part_1')
folder_2 = os.listdir('/content/ham10000_images_part_2')

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

y_train = []
y_valid = []

# Transfer the training images
for image in train_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join('ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

# Transfer the validation images
for image in val_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join('ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)


#--------------image generator---------------
from keras.preprocessing.image import ImageDataGenerator
import keras 
print(df.head())
image_class = ['nv','mel','bkl','bcc','akiec','vasc','df']

train_path = 'base_dir/train_dir/'
valid_path = 'base_dir/val_dir/'
print(os.listdir('base_dir/train_dir'))
print(len(os.listdir('base_dir/val_dir')))

image_shape = 224


train_datagen  = ImageDataGenerator(rescale=1./255)
val_datagen  = ImageDataGenerator(rescale=1./255)


train_batches = train_datagen.flow_from_directory(train_path, 
                                                        target_size = (image_shape,image_shape),
                                                        classes = image_class,
                                                        batch_size = 64
                                                        )

valid_batches = val_datagen.flow_from_directory(valid_path, 
                                                        target_size = (image_shape,image_shape),
                                                        classes = image_class,
                                                        batch_size = 64
                                                      
                                                        )


#from sklearn.preprocessing import LabelEncoder
'''
Encoder_X = LabelEncoder()
for col in df.columns:
  if df.dtypes[col] == 'object':
    df[col] = col + '_' + df[col].map(str)
    df[col] = Encoder_X.fit_transform(self, df[col])
y = df['dx']
'''

#transforms labels into vectors using one hot encoder
'''from numpy import array 
from keras.utils import to_categorical
print(y_train)
y = array(Y_train)
y = to_categorical(y)
print(y)'''

#----------------model------------
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten 
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model


num_train_samples = 9013
num_val_samples = 1002
num_classes = 7

# Mobile Net 
mobile = keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
# Add a dropout and dense layer for predictions
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
print(mobile.input)
net = Model(inputs=mobile.input, outputs=predictions)

mobile.summary()

#compile model
for layer in net.layers[:-23]:
    layer.trainable = False
net.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

history = net.fit_generator(train_batches, epochs=30)

#outputs 70%
acc_valid = net.evaluate_generator(valid_batches)
print(acc_valid)

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
predictions = net.predict_generator(valid_batches,verbose=1)
test_labels = valid_batches.classes
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
predfinal = predictions.argmax(axis=1)
cm = confusion_matrix(test_labels, predfinal)

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels)
