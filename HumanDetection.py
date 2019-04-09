# Databricks notebook source
# DBTITLE 1,Importing Libraries
import numpy as np
import os
import time

from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten


from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


# COMMAND ----------

# DBTITLE 1,Importing VGG16 Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input
shape  = Input(shape=(224,224,3))
model1 = VGG16(include_top=False,input_tensor=shape)
model1.summary()

# COMMAND ----------

# DBTITLE 1,Creating second model with 1 layer
from keras.models import Sequential
from keras.layers import Dense
model2 = Sequential()
model2.add(Dense(2, activation='softmax'))

# COMMAND ----------

# DBTITLE 1,Merging both models
from keras.models import Sequential
from keras.layers import Dense
newmodel = Sequential()
newmodel.add(model1)
newmodel.add(model2)
newmodel.add(Flatten())
newmodel.add(Dense(2, activation='softmax'))
newmodel.summary()

# COMMAND ----------

# DBTITLE 1,Extracting tar file
"""
import tarfile
tf = tarfile.open("/dbfs/FileStore/tables/INRIAPerson.tar")
tf.extractall()
"""
# COMMAND ----------

# DBTITLE 1,Preprocessing & Feature Generation
path = "/databricks/driver"
data_path_test = path+"/INRIAPerson/Test"
data_path_train = path+"/INRIAPerson/Train"
data_dir_list_test = os.listdir(data_path_test)
data_dir_list_train = os.listdir(data_path_train)
data_path = ""

# COMMAND ----------

for val in data_dir_list_train:
        path = data_path_train+'/'+val
        print(path)
data_path = ['/Test/pos','/Test/neg']

# COMMAND ----------

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
img_data_list = []
count = 0
for dataset in data_path:
  img_ls = os.listdir("/databricks/driver/INRIAPerson"+dataset+"/")
  for val in img_ls:
        path = "/databricks/driver/INRIAPerson"+dataset+"/"+val
        img = image.load_img(path,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        x = x/255
        print('Input Image Shape: ',x.shape)
        img_data_list.append(x)
        count += 1
  if count==1500:
    break

# COMMAND ----------

len(img_data_list)

# COMMAND ----------

import numpy as np
img_data = np.array(img_data_list)
img_data = img_data.reshape(741,224, 224, 3)
img_data.shape

# COMMAND ----------

num_classes = 2
num_of_samples = img_data.shape[0]
lp = [1]*400
ln = [0]*341
labels = np.array(lp+ln)
labels.shape

# COMMAND ----------

# DBTITLE 1,Encoding
from keras.utils import np_utils
targets = np_utils.to_categorical(labels,num_classes)
targets

# COMMAND ----------

# DBTITLE 1,Splitting Dataset
from sklearn.utils import shuffle
#Shuffle the dataset
x,y = shuffle(img_data,targets,random_state=2)
# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# COMMAND ----------

# DBTITLE 1,Model Compilation,Fitting & Evaluation
last_layer = model1.get_layer('block5_pool').output
for layer in newmodel.layers[:-1]:
	layer.trainable = False

newmodel.layers[-1].trainable
newmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = newmodel.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = newmodel.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))



