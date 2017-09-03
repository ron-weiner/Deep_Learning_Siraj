
# coding: utf-8

# In[1]:

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# In[2]:

import pandas as pd


# # Load IMDB Dataset

# In[3]:

train, test, _ = imdb.load_data(path='imdb.pkl',
                                n_words=10000,
                                valid_portion=0.1) # 10% of data as "validation set"


# In[4]:

trainX, trainY = train
testX,  testY  = test


# ### All about trainX

# In[5]:

pd.Series(trainX).tail()


# In[6]:

print( list(pd.Series(trainX).iloc[5555]) )


# In[7]:

pd.Series(trainX).shape


# ### All about trainY

# In[8]:

pd.Series(trainY).tail()


# In[9]:

pd.Series(trainY).shape


# In[10]:

pd.Series(trainY).value_counts()


# In[11]:

pd.Series(trainY).value_counts().index.tolist()


# In[12]:

len(pd.Series(trainY).value_counts().index.tolist())


# # Data Preprocessing

# ### Sequence Padding
# 
# Pad each sequence to the same length: the length of the longest sequence.
# If maxlen is provided, any sequence longer than maxlen is truncated to
# maxlen. Truncation happens off either the beginning (default) or the
# end of the sequence. Supports post-padding and pre-padding (default).

# In[13]:

trainX = pad_sequences(trainX, maxlen=100, value=0.0)
testX  = pad_sequences(testX,  maxlen=100, value=0.0)


# In[14]:

trainX.shape


# In[15]:

pd.DataFrame(trainX).tail()


# In[16]:

pd.DataFrame(testX).tail()


# ### Convert Labels to Vectors
# Converting labels to binary vectors

# In[17]:

trainY = to_categorical(trainY, nb_classes=2)
testY  = to_categorical(testY,  nb_classes=2)


# In[18]:

trainY


# In[19]:

pd.DataFrame(trainY).tail()


# # Network Building

# In[20]:

# The first element is the "batch size" which we set to "None"
# The second element is set to "100" coz we set the max sequence length to "100"
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128) # input_dim: Vocabulary size (number of ids)
net = tflearn.lstm(net, 128, dropout=0.8) # Long Short Term Memory Recurrent Layer
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, 
                         optimizer='adam', 
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


# # Training

# In[21]:

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)


# In[ ]:



