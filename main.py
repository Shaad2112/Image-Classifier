import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

data_dir='Image classifier/data'
image_exts=['jpeg','jpg','bmp','png']
#print(os.listdir(os.path.join(data_dir, 'happy')))

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path=os.path.join(data_dir,image_class,image)
        try:
            img=cv2.imread(image_path)
            tip=imghdr.what(image_path)
            if tip not in image_exts:
                print('image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
                print('issue with image{}'.format(image_path))


data=tf.keras.utils.image_dataset_from_directory('Image classifier/data') #reshaps automatically
data_iterator=data.as_numpy_iterator()
batch=data_iterator.next()
#print(batch)
#class 1 is sad 
#class 2 is happy
'''fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()'''

data=data.map(lambda x,y:(x/255,y))

train_size=int(len(data)*.7)
val_size=int(len(data)*.2)+1
test_size=int(len(data)*.1)+1

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)

model=Sequential()
model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
print(model.summary())

logdir='Image classifier/log'
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist=model.fit(train,epochs=15,validation_data=val,callbacks=[tensorboard_callback])

#plot
fig=plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('loss',fontsize=20)
plt.legend(loc="upper left")

fig=plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuraccy')
fig.suptitle('accuracy',fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre=Precision()
re=Recall()
acc=BinaryAccuracy()
for batch in test.as_numpy_iterator():
     x,y=batch
     yhat=model.predict(x)
     pre.update_state(y,yhat)
     re.update_state(y,yhat)
     acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')

img=cv2.imread('Image classifier/sad.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()



yhat=model.predict(np.expand_dims(resize/255,0))
print(yhat)

if yhat >0.5:
     print('predicted class is sad')
else:
     print('predicted class is happy')

model.save(os.path.join('Image classifier/models','happySADmodel.h5'))
