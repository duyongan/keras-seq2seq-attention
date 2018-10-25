from keras.layers import Embedding, GRU, Input, Reshape, Dense
from keras.models import Model
from keras.layers.core import Reshape,Lambda,Flatten
from keras.layers.wrappers import Bidirectional

lenth_of_seq=10
vocab_size=5000
out_length=10
embedding_size=300
GRU_cell=100
Dense_cell=out_length*100

x = Input(shape=(lenth_of_seq,))
x_embedding = Embedding(vocab_size, embedding_size, input_length=lenth_of_seq)(x)
hidden_layer = Bidirectional(GRU(GRU_cell,return_sequences=True))(x_embedding)
mid=Flatten()(hidden_layer)
mid = Dense(Dense_cell,use_bias=False)(mid)
mid=Reshape((out_length,-1))(mid)
mid = Bidirectional(GRU(GRU_cell,return_sequences=True))(mid)
y=Dense(vocab_size,activation='softmax')(mid)
model = Model(inputs=x, outputs=y)
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['acc'])
model.summary()
