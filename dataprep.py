from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Conv2D
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.preprocessing.image import ImageDataGenerator
import time
from IPython.display import display 
#from PIL import Image
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras
from PIL import Image, ImageFilter
from keras import backend as k
config = tf.ConfigProto()
    
#BASE_DIR='/tmp'
S3_DATA_BUCKET_NAME='autonomous-mobility'
DATASET_NAME='training-data'
train_data_dir = DATASET_NAME
#train_data_dir = BASE_DIR+'/'+DATASET_NAME
validation_data_dir = DATASET_NAME
#validation_data_dir = BASE_DIR+'/'+DATASET_NAME

!aws iam get-user

!aws s3 sync s3://$S3_DATA_BUCKET_NAME/$DATASET_NAME $DATASET_NAME

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
starttime = time.time()

# dimensions of our images.
img_width, img_height = 180,320#720, 1280#180, 320


nb_train_samples = 10
nb_validation_samples = 100
nb_epoch = 1
batch_size = 200


from keras.models import model_from_json
import numpy as np
import os
import cv2
import skimage 
from skimage import io
import pandas as pd
import pickle as pk
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
flag = 1

imagesName = os.listdir(train_data_dir)

print('No of images found in this directory', len(imagesName))
count = 0
filecount = 0
tempImages = []
for image in imagesName:
    cap = io.imread(train_data_dir + "/" +image)
    cap = rescale(cap, .25, anti_aliasing=False)
    tempImages.append(cap)
    count = count + 1
    if(count == 100):
       count = 0
       tempImages = np.array(tempImages)
       tempImages = tempImages.reshape(tempImages.shape[0],-1)
       filename = "pickle"+str(filecount)+".pk"
       file = open("pickle/"+filename,"wb")
       pk.dump(tempImages,file)
       filecount = filecount + 1
       tempImages = []
       print("pikle....")
       file.close()
        
print('Finished!')     