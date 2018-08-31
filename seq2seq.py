import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.layers import Embedding, TimeDistributed, RepeatVector,concatenate , GRU, Input, Reshape, Dense, Flatten
from keras.models import Model,load_model
from keras.layers.core import Reshape


lenth_of_seq=20
vocab_size=4000
x = Input(shape=(lenth_of_seq,))
x_embedding = Embedding(vocab_size, 50, input_length=lenth_of_seq)(x)
hidden_layer = GRU(100,return_sequences=True)(x_embedding)
mid_list=[]
for i in range(lenth_of_seq):
    mid = Dense(5)(hidden_layer)
    mid = Reshape((1,100))(mid)
    mid_list.append(mid)
mid_2=concatenate(mid_list,axis=1)
y = GRU(100,return_sequences=True)(mid_2)
model = Model(inputs=x, outputs=y)
y=Dense(5,activation='softmax')(y)
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['acc'])
model.summary()
