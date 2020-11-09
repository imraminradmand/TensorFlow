#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('../DATA/cancer_classification.csv')


# In[3]:


df.info()


# No null values 

# In[5]:


df.describe().transpose()


# In[6]:


sns.countplot(x='benign_0__mal_1', data=df)


# relatively well balanced dataset

# In[10]:


df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# checking corrolation, excluding the pram itself

# In[13]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr())


# In[14]:


X = df.drop('benign_0__mal_1', axis=1).values
y= df['benign_0__mal_1'].values


# In[15]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler()


# In[39]:


X_train = scaler.fit_transform(X_train)


# In[40]:


X_test = scaler.transform(X_test)


# not fitting to test set to prevent data leakage

# In[41]:


import tensorflow as tf


# In[42]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


# In[43]:


X_train.shape


# In[44]:


model = Sequential()

model.add(Dense(30, activation='relu'))

model.add(Dense(15, activation='relu'))

#Binary classification should use sigmoid instead of relu
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')


# In[45]:


model.fit(x = X_train, y = y_train, epochs=600, validation_data=(X_test, y_test))


# In[46]:


losses = pd.DataFrame(model.history.history)


# In[47]:


losses.plot()


# Clearly model was overfitted

# In[48]:


model = Sequential()

model.add(Dense(30, activation='relu'))

model.add(Dense(15, activation='relu'))

#Binary classification should use sigmoid instead of relu
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')


# re-defining model

# In[49]:


from tensorflow.keras.callbacks import EarlyStopping


# In[50]:


help(EarlyStopping)


# In[52]:


early_stop = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=25)


# monitoring val loss in min mode to minimize val loss

# In[53]:


model.fit(x = X_train, y = y_train, epochs=600, validation_data=(X_test, y_test),
         callbacks=[early_stop])


# In[54]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# early stop did improve val loss

# In[55]:


model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

#Binary classification should use sigmoid instead of relu
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')


# re-defining model with dropout to possibly better model=

# In[56]:


model.fit(x = X_train, y = y_train, epochs=600, validation_data=(X_test, y_test),
         callbacks=[early_stop])


# In[57]:


model_loss = pd.DataFrame(model.history.history)


# In[58]:


model_loss.plot()


# more improvement seen

# In[60]:


predictions = model.predict_classes(X_test)


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix


# In[62]:


print(classification_report(y_test, predictions))


# In[63]:


print(confusion_matrix(y_test, predictions))


# Decent outcome

# In[64]:


from tensorflow.keras.models import load_model


# In[65]:


model.save('Cancer_Predictor.h5')


# In[ ]:




