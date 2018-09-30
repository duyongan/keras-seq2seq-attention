import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.layers import Embedding, TimeDistributed, RepeatVector,concatenate , GRU, Input, Reshape, Dense, Flatten
from keras.models import Model,load_model
from keras.layers.core import Reshape
from keras.layers.wrappers import Bidirectional


lenth_of_seq=len(xs[0])
vocab_size=len(tokenizer.word_index)+1
out_length=len(ys[0])

x = Input(shape=(lenth_of_seq,))
x_embedding = Embedding(vocab_size, 100, input_length=lenth_of_seq)(x)
hidden_layer = Bidirectional(GRU(100,return_sequences=True))(x_embedding)
mid_list=[]
for i in range(out_length):
    mid = Dense(5)(hidden_layer)
    mid = Reshape((1,lenth_of_seq*5))(mid)
    mid_list.append(mid)
mid_2=concatenate(mid_list,axis=1)
y = Bidirectional(GRU(100,return_sequences=True))(mid_2)
y=Dense(vocab_size,activation='softmax')(y)
model = Model(inputs=x, outputs=y)
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['acc'])
model.summary()
