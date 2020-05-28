#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[2]:


mit_test_data = pd.read_csv("Dataset/mitbih_test.csv", header=None)
mit_train_data = pd.read_csv("Dataset/mitbih_train.csv", header=None)


# In[6]:



X = mit_train_data.loc[:, mit_train_data.columns != 187]

y = mit_train_data.loc[:, mit_train_data.columns == 187]
y = to_categorical(y)

testX = mit_test_data.loc[:, mit_test_data.columns != 187]

testy = mit_test_data.loc[:, mit_test_data.columns == 187]
testy = to_categorical(testy)


# In[7]:


model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(187,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=100)

print("Evaluation: ")
mse, acc = model.evaluate(testX, testy)
print('mean_squared_error :', mse)
print('accuracy:', acc)


# In[8]:


model.save('my_model')


# In[ ]:




