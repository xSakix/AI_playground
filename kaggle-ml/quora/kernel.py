# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
np.set_printoptions(threshold=np.nan)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))

# Any results you write to the current directory are saved as output.

import gensim
from gensim.utils import simple_preprocess
from keras.preprocessing.sequence import pad_sequences


def load_x_from_df(df,model):
    sequences = []
    for question_text in df['question_text'].values:
        tokens = simple_preprocess(question_text)
        sentence = []
        for word in tokens:
            # print(model.wv[word])
            if word in model.wv.vocab:
                sentence.append(model.wv[word])
        if len(sentence) == 0:
            sentence = np.zeros((max_len,300))
        sequences.append(np.mean(sentence,axis=1))

    x = pad_sequences(sequences,dtype='float32', maxlen=max_len)
    x = x.reshape(x.shape[0],1,x.shape[1])
    return x



print('loading word2vec model...')
model = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)

df = pd.read_csv('../input/train.csv')

print('columns:',df.columns)
pd.set_option('display.max_columns',None)
print('df head:',df.head())
print('example of the question text values:',df['question_text'].head().values)
print('what values contains target:',df.target.unique())

print('loading sequences...')
max_len = df['question_text'].apply(lambda x:len(x)).max()
print('max length of sequences:',max_len)

print('creating sequences')
x = load_x_from_df(df,model)
print(x.shape)
y = df.target.values
print(y.shape)

print('Creating model...')
#inpiration from : https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Masking(input_shape=(x.shape[1],x.shape[2])))
model.add(GlobalAveragePooling1D())
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

print('fiting model...')
history = model.fit(x,y,validation_split=0.2,epochs=100, callbacks=[EarlyStopping(patience=20)])

print('model score:',model.evaluate(x,y))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()