
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import theano
from keras import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# In[30]:


wine = pd.read_csv('/Users/ajay/Downloads/Python_ML/winequality-red.csv')


# In[32]:


num_out = wine['quality'].nunique()
batch_size = 3
num_out




# In[33]:


def reset(x):
       return x-3
    

Y = wine['quality'].apply(reset).values


X = wine.drop(['quality'],axis=1).values



# In[34]:







X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)










# In[35]:


Y_train = keras.utils.to_categorical(Y_train, num_out)
Y_test = keras.utils.to_categorical(Y_test, num_out)

Y_train.shape



# In[37]:


def ulti_model():
    model1 = Sequential()
    model1.add(Dense(22, activation='relu', input_dim = 11))

    model1.add(Dense(22, activation='relu'))

    model1.add(Dense(6, activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model1


# In[39]:


model = Sequential()
model.add(Dense(22, activation='relu', input_dim = 11))

model.add(Dense(22, activation='relu'))

model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[40]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[41]:


history = model.fit(X_train, Y_train,batch_size=batch_size,epochs=50,validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


Y_train.shape


# In[ ]:


X_train.shape


# In[ ]:


# NN AFTER GRID SEARCH 


# In[ ]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# In[ ]:


param_grid = dict(epochs=range(1,50),batch_size=range(1,50))


# In[42]:


param_grid


# In[43]:


model1 = KerasClassifier(build_fn=ulti_model,epochs=10)


# In[46]:


grid = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1)


# In[ ]:


grid_result = grid.fit(X_train, Y_train)


# In[ ]:


grid.best_params_

