
# coding: utf-8

# # Siraj's 03 Challenge
# 
# ### This is a response to the Coding Challenge in: [https://youtu.be/si8zZHkufRY](https://youtu.be/si8zZHkufRY?t=9m4s)
# 
# > The challenge for this video is to train a model on this dataset of video game reviews from IGN.com. Then, given some new video game title it should be able to classify it. You can use pandas to parse this dataset. Right now each review has a label that's either Amazing, Great, Good, Mediocre, Painful, or Awful. These are the emotions. Using the existing labels is extra credit. The baseline is that you can just convert the labels so that there are only 2 emotions (positive or negative). Ideally you can use an RNN via TFLearn like the one in this example, but I'll accept other types of ML models as well. You'll learn how to parse data, select appropriate features, and use a neural net on an IRL problem. 

# # Sentiment Labels to be Predicted
# 
# - Great          
# - Good           
# - Okay           
# - Mediocre       
# - Amazing        
# - Bad            
# - Awful          
# - Painful        
# - Unbearable     
# - Masterpiece    

# # Accuracy Results 
# - `Dummy Classifier (i.e. select most frequent class): 0.25631 (25.6%)`
# - `Multinomial Naive Bayes:                            0.32355 (32.4%)`
# - `RNN (using tflearn):                                0.41546 (41.5%)`

# In[1]:

import sys
import tensorflow as tf
from termcolor import colored
print(colored('Python Version: %s' % sys.version.split()[0], 'blue'))
print(colored('TensorFlow Ver: %s' % tf.__version__, 'magenta'))


# In[2]:

n_epoch = int(input('Enter no. of epochs for RNN training: '))


# In[3]:

print(colored('No. of epochs: %d' % n_epoch, 'red'))


# In[4]:

import pandas as pd
pd.set_option('display.max_colwidth', 1000)


# # Load IGN Dataset as `original_ign`

# In[5]:

original_ign = pd.read_csv('ign.csv')
original_ign.head(10)


# ### Check out the `shape` of the IGN Dataset

# In[6]:

print('original_ign.shape:', original_ign.shape)


# ### Check out all the unique `score_phrase` as well as their `counts`

# In[7]:

original_ign.score_phrase.value_counts()


# # Data Preprocessing
# 
# As always, we gotta perform preprocessing on our Dataset before training our model(s).

# ### Convert `score_phrase` to binary sentiments and add a new column called `sentiment`

# In[8]:

bad_phrases = ['Bad', 'Awful', 'Painful', 'Unbearable', 'Disaster']
original_ign['sentiment'] = original_ign.score_phrase.isin(bad_phrases).map({True: 'Negative', False: 'Positive'})


# In[9]:

# Remove "Disaster"
original_ign = original_ign[original_ign['score_phrase'] != 'Disaster']


# In[10]:

original_ign.head()


# ### No. of Positive Sentiments VS No. of Negative Seniments

# In[11]:

original_ign.sentiment.value_counts(normalize=True)


# ### Check for null elements

# In[12]:

original_ign.isnull().sum()


# ### Fill all null elements with an empty string

# In[13]:

original_ign.fillna(value='', inplace=True)


# In[14]:

# original_ign[ original_ign['genre'] == '' ].shape


# # Create a new DataFrame called `ign`

# In[15]:

ign = original_ign[ ['sentiment', 'score_phrase', 'title', 'platform', 'genre', 'editors_choice'] ].copy()
ign.head(10)


# ### Create a new column called `is_editors_choice`

# In[16]:

ign['is_editors_choice'] = ign['editors_choice'].map({'Y': 'editors_choice', 'N': ''})
ign.head()


# ### Create a new column called `text` which contains contents of several columns

# In[17]:

ign['text'] = ign['title'].str.cat(ign['platform'], sep=' ').str.cat(ign['genre'], sep=' ').str.cat(ign['is_editors_choice'], sep=' ')


# In[18]:

print('Shape of \"ign\" DataFrame:', ign.shape)


# In[19]:

ign.head(10)


# ![http://www.westernbands.net/userdata/news_picupload/pic_sid1070-0-norm.jpg](http://www.westernbands.net/userdata/news_picupload/pic_sid1070-0-norm.jpg)

# # Here, I'll treat this as a `multiclass problem` where I attempt to predict the labels (i.e. the `score_phrases`)
# 
# Examples of **score_phrases**: 
# - Great        
# - Good         
# - Okay        
# - Mediocre    
# - Amazing      
# - Bad        
# - Awful        
# - Painful      
# - Unbearable      
# - Masterpiece     

# In[20]:

X = ign.text
y = ign.score_phrase


# ### Top 10 rows for `X`

# In[21]:

X.head(10)


# ### Top 10 rows for `y`

# In[22]:

y.head(10)


# # Model #0: The DUMMY Classifier (Always Choose the *Most Frequent* Class)

# In[23]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# In[24]:

vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b')
dummy = DummyClassifier(strategy='most_frequent', random_state=0)

dummy_pipeline = make_pipeline(vect, dummy)


# In[25]:

dummy_pipeline.named_steps


# In[26]:

# Cross Validation
cv = cross_val_score(dummy_pipeline, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(colored('\nDummy Classifier\'s Accuracy: %0.5f\n' % cv.mean(), 'yellow'))


# # Model #1: MultinomialNB Classifier

# In[27]:

from sklearn.naive_bayes import MultinomialNB


# In[28]:

vect = TfidfVectorizer(stop_words='english', 
                       token_pattern=r'\b\w{2,}\b',
                       min_df=1, max_df=0.1,
                       ngram_range=(1,2))
mnb = MultinomialNB(alpha=2)

mnb_pipeline = make_pipeline(vect, mnb)


# In[29]:

mnb_pipeline.named_steps


# In[30]:

# Cross Validation
cv = cross_val_score(mnb_pipeline, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(colored('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % cv.mean(), 'green'))


# # Model #2: RNN Classifier using `TFLearn`

# In[31]:

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# ### Train-Test-Split

# In[32]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ### Create the `vocab` (so that we can create `X_word_ids` from `X`)

# In[33]:

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')


# In[34]:

vect.fit(X_train)
vocab = vect.vocabulary_


# In[35]:

def convert_X_to_X_word_ids(X):
    return X.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab] )


# In[36]:

X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids  = convert_X_to_X_word_ids(X_test)


# ### Difference between X(_train/_test) and X(_train_word_ids/test_word_ids)

# In[37]:

X_train.head()


# In[38]:

X_train_word_ids.head()


# In[39]:

print('X_train_word_ids.shape:', X_train_word_ids.shape)
print('X_test_word_ids.shape:', X_test_word_ids.shape)


# ### Sequence Padding

# In[40]:

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=20, value=0)
X_test_padded_seqs  = pad_sequences(X_test_word_ids , maxlen=20, value=0)


# In[41]:

print('X_train_padded_seqs.shape:', X_train_padded_seqs.shape)
print('X_test_padded_seqs.shape:', X_test_padded_seqs.shape)


# In[42]:

pd.DataFrame(X_train_padded_seqs).head()


# In[43]:

pd.DataFrame(X_test_padded_seqs).head()


# ### Convert (y) labels to vectors

# In[44]:

unique_y_labels = list(y_train.value_counts().index)
unique_y_labels


# In[45]:

len(unique_y_labels)


# In[46]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(unique_y_labels)


# In[47]:

# print('')
# print(unique_y_labels)
# print(le.transform(unique_y_labels))
# print('')


# In[48]:

print('')
for label_id, label_name in zip(le.transform(unique_y_labels), unique_y_labels):
    print('%d: %s' % (label_id, label_name))
print('')


# In[49]:

y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))
y_test  = to_categorical(y_test.map(lambda x:  le.transform([x])[0]), nb_classes=len(unique_y_labels))


# In[50]:

y_train[0:3]


# In[51]:

print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)


# ### Network Building

# In[52]:

size_of_each_vector = X_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)


# In[53]:

print('size_of_each_vector:', size_of_each_vector)
print('vocab_size:', vocab_size)
print('no_of_unique_y_labels:', no_of_unique_y_labels)


# In[54]:

#sgd = tflearn.SGD(learning_rate=1e-4, lr_decay=0.96, decay_step=1000)

net = tflearn.input_data([None, size_of_each_vector]) # The first element is the "batch size" which we set to "None"
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128) # input_dim: vocabulary size
net = tflearn.lstm(net, 128, dropout=0.6) # Set the dropout to 0.6
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax') # relu or softmax
net = tflearn.regression(net, 
                         optimizer='adam',  # adam or ada or adagrad # sgd
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


# ### Intialize the Model

# In[55]:

#model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='SavedModels/model.tfl.ckpt')
model = tflearn.DNN(net, tensorboard_verbose=0)


# ### Train the Model

# In[56]:

# model.fit(X_train_padded_seqs, y_train, 
#           validation_set=(X_test_padded_seqs, y_test), 
#           n_epoch=n_epoch,
#           show_metric=True, 
#           batch_size=100)


# ### Manually Save the Model

# In[57]:

# model.save('SavedModels/model.tfl')
# print(colored('Model Saved!', 'red'))


# ### Manually Load the Model

# In[58]:

model.load('SavedModels/model.tfl')
print(colored('Model Loaded!', 'red'))


# ### RNN's Accuracy

# In[59]:

import numpy as np
from sklearn import metrics


# In[60]:

pred_classes = [np.argmax(i) for i in model.predict(X_test_padded_seqs)]
true_classes = [np.argmax(i) for i in y_test]

print(colored('\nRNN Classifier\'s Accuracy: %0.5f\n' % metrics.accuracy_score(true_classes, pred_classes), 'cyan'))


# -------------

# ### Show some predicted samples

# In[61]:

ids_of_titles = range(0,21) # range(X_test.shape[0]) 

for i in ids_of_titles:
    pred_class = np.argmax(model.predict([X_test_padded_seqs[i]]))
    true_class = np.argmax(y_test[i])
    
    print(X_test.values[i])
    print('pred_class:', le.inverse_transform(pred_class))
    print('true_class:', le.inverse_transform(true_class))
    print('')


# -----------------------

# By Jovian Lin ([http://jovianlin.com](http://jovianlin.com))
