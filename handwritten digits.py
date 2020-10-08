
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.logging.set_verbosity(tf.logging.ERROR)


# In[18]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[19]:


from tensorflow.keras.utils import to_categorical


# In[20]:


y_train_1 = to_categorical(y_train)
y_test_1 = to_categorical(y_test)


# In[21]:


x_train_2 = np.reshape(x_train,(60000,784))
x_test_2 = np.reshape(x_test, (10000,784))


# In[22]:


x_mean = np.mean(x_train_2)
x_std = np.std(x_train_2)


# In[23]:


epsilon = 1e-10
x_norm = (x_train_2 - x_mean)/(x_std +epsilon)
x_testnorm = (x_test_2 - x_mean)/(x_std +epsilon)


# In[8]:


model = Sequential([
    Dense(128, activation = 'relu',input_shape = (784,)),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

model.compile(
optimizer = 'sgd',
loss = 'categorical_crossentropy',
metrics = ['accuracy'])

model.summary()


# In[9]:


model.fit(x_norm, y_train_1, epochs = 4)


# In[13]:


_, accuracy = model.evaluate(x_testnorm,y_test_1 )
print("Accuracy:",accuracy*100)


# In[32]:


preds = model.predict(x_testnorm)
plt.figure(figsize=(12,12))
start = 10

for i in range(5):
    plt.subplot(5,5,i+1)
    plt.grid = False
    plt.xticks = []
    plt.yticks = []
    
    pred = np.argmax(preds[start + i])
    gt = y_test[start + i]
    col = 'g'
    if pred != gt:
        col = 'r'
    plt.xlabel('i={},pred={},gt={}'.format(start+i,pred,gt), color = col)
    plt.imshow(x_test[start+i],cmap = 'binary')

plt.show()
    

