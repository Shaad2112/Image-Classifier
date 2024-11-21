import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


img=cv2.imread('Image classifier/sad.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

new_model=load_model(os.path.join('models','happySADmodel.h5'))
yhat=new_model.predict(np.expand_dims(resize/255,0))


if yhat >0.5:
     print('predicted class is sad')
else:
     print('predicted class is happy')


