# _code_for_Training_the_model...

import tensorflow 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from xgboost import XGBClassifier
import shutil
import glob
import re
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.io import imread, imshow


data = 'C:/Users/YK/PycharmProjects/Py/Func_tumor_seg/CT brain dataset'
No_brain_tumor = 'C:/Users/YK/PycharmProjects/Py/Func_tumor_seg/CT brain dataset/no'
Yes_brain_tumor = 'C:/Users/YK/PycharmProjects/Py/Func_tumor_seg/CT brain dataset/yes'

dirlist = [No_brain_tumor, Yes_brain_tumor]
# ['brain_tumor_dataset/no/', 'brain_tumor_dataset/yes/']
classes = [0, 1]
filepaths = []
labels = []
for i, j in zip(dirlist, classes):
    # i
    # ../input/brain-mri-images-for-brain-tumor-detection/no/
    # ../input/brain-mri-images-for-brain-tumor-detection/yes/
    # j
    # No
    # Yes

    filelist = os.listdir(i)
    print(filelist)
    print('\n')
    # os.listdir --> returns a list containing the names of the entries in the directory given by path.
    for f in filelist:
        filepath = os.path.join(i, f)
        # os.path.join('brain_tumor_dataset/no/','1 no.jpeg;)
        # brain_tumor_dataset/no/1 no.jpeg
        filepaths.append(filepath)
        # store the path into empty list called filepaths
        labels.append(j)
    print(filepaths)
    print('\n')
    print(labels)
    print('\n')
print('filepaths: ', len(filepaths), '   labels: ', len(labels))

Files = pd.Series(filepaths, name='filepaths')
Label = pd.Series(labels, name='labels')
df = pd.concat([Files, Label], axis=1)
# df=pd.DataFrame(np.array(df).reshape(253,2), columns = ['filepaths', 'labels'])
# df.head()
print(df)

for i in df['filepaths']:
    print(i)

plt.figure(figsize=(4, 4))
for i in range(0, 10):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = mpimg.imread(df['filepaths'][i])
    img_name = re.sub(r'^\D+', '', df['filepaths'][i])

    # ax.imshow(img)
    # ax.set_title(img_name)


from PIL import Image

widths = []
heights = []
for idx, row in df.iterrows():
    path = row['filepaths']

    #   print(path)
    im = Image.open(path)

    #     print(im)
    # <PIL.JpegImagePlugin.JpegImageFile image mode=L size=630x630 at 0x1888E54F7C0>
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=173x201 at 0x1888E59BFA0>
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x1888E540340>
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=275x183 at 0x1888D29BFA0>
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x1888E540C70>
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=177x197 at 0x1888E5317F0>
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=232x217 at 0x1888E540C70>
    #     print(im.size)
    # (630, 630)
    # (173, 201)
    # (300, 168)
    # (275, 183)
    # (300, 168)
    width, height = im.size
    widths.append(width)
    heights.append(height)
avg_width = int(sum(widths) / len(widths))
avg_height = int(sum(heights) / len(heights))
print(avg_width, avg_height)

from tensorflow.keras.preprocessing.image import load_img


# Image Resize Function
def load_resize_color_image(path):
    # load image and resize to 300x300
    image = load_img(path, target_size=(100, 100))

    return image



image_list = []
cancer_list = []

from tensorflow.keras.preprocessing.image import img_to_array

for idx, row in df.iterrows():
    path = row['filepaths']
    cancer = row['labels']
    # print(path)
    # print(cancer)
    # C: / Users / YK / PycharmProjects / Py / Func_tumor_seg / CTbrain dataset / no\no997.jpg
    # 0
    # C: / Users / YK / PycharmProjects / Py / Func_tumor_seg / CT brain dataset / no\no998.jpg
    # 0
    # C: / Users / YK / PycharmProjects / Py / Func_tumor_seg / CT brain dataset / yes\y1.jpg
    # 1
    image = load_resize_color_image(path)
    # turn image to array
    image_array = img_to_array(image)
    image_list.append(image_array)
    cancer_list.append(cancer)
# print(image_list[0:5])
# #
# print(cancer_list[0:5])

# image_list = image_list[::2]
# cancer_list = cancer_list[::2]
from sklearn.utils import shuffle

image_list, cancer_list = shuffle(image_list, cancer_list)

X_data = np.array(image_list)
y_data = np.array(cancer_list)
print(X_data.shape)
print(y_data.shape)

from sklearn.model_selection import train_test_split

X_train,  X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=0)
print(X_train.shape)
print(y_train.shape)

X_train = X_train / 255
X_test = X_test / 255
print(X_test.shape)
print(y_test.shape)


print(y_train)
print(y_test)

from tensorflow.keras.layers import LeakyReLU
from keras.layers import Layer
from keras import backend as K
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D, MaxPool2D
from tensorflow.keras.models import Sequential, Model

epochs = 20
batch_size = 10
input_shape = (100, 100, 3)
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(Flatten())
# CNN structure
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()




from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop

#   define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy
model.compile(loss="binary_crossentropy",
              optimizer='Adam',
              metrics=['accuracy'])
# training
# noinspection PyArgumentList
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test,y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
acc_cnn = score[1]
print(acc_cnn)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# print(y_train)
# print(X_test)

#interlayer_layer_model for hybrid model construction

from tensorflow.keras.models import Model
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.output)


from sklearn import preprocessing
from sklearn import utils

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# print(intermediate_output[::10])
# print(intermediate_test_output)

# print(intermediate_output[0])


intermediate_output = intermediate_layer_model.predict(X_train)
submission_cnn = model.predict(X_test)
intermediate_test_output = intermediate_layer_model.predict(X_test)

intermediate_layer_model.summary()

print(intermediate_output.shape)
print(intermediate_test_output.shape)

x = intermediate_output
x1 = intermediate_test_output
print(intermediate_output.shape)
print(y_train.shape)

#XGBoost model definition
xgb = XGBClassifier(n_estimators=100)
training_start = time.perf_counter()
xgb.fit(intermediate_output, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(x1)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))

from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,preds,target_names=["proposed framework","proposed framework"]))


#SVM model definition
model1 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
training_start = time.perf_counter()
h = model1.fit(intermediate_output, y_train)
preds = model1.predict(x1)
# print('conv-SVM', preds)
acc_svc = (preds == y_test).sum().astype(float) / len(preds) * 100
print("conv-SVM %3.2f" % (acc_svc))
print("Model Accuracy:")
fin = {"Accuracy":(acc_cnn*100+acc_xgb+acc_svc)/3}
print(fin)


score = accuracy_score(y_test, preds)
print("accuracy:   %0.3f" % score)

#   Train the model and test/validate the mode with the test data after each cycle (epoch) through the training data
#   Return history of loss and accuracy for each epoch


# import joblib
# #
# # # Save the model as a pickle in a file
# joblib.dump(model1, 'D:/Conv-SVM.pkl')
# #
# # # Save the model as a pickle in a file
# joblib.dump(xgb, 'D:/Conv-XGB.pkl')
# #
# # #
# model.save("D:/conv-SVM.h5")
# #
