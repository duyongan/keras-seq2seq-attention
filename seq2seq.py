from keras.layers import Embedding, TimeDistributed, RepeatVector,concatenate , GRU, Input, Reshape, Dense, Flatten,LSTM,Permute
from keras.models import Model,load_model
from keras.layers.core import Reshape,Lambda,Flatten
from keras.layers.wrappers import Bidirectional

lenth_of_seq=10
vocab_size=5000
out_length=5
embedding_size=300
GRU_cell=100
Dense_cell=out_length*100

#版本一：铺平做所有维度的attention
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

#版本二：做两个词向量同维度上的attention
#例如input:[x1,x2,x3],[x4,x5,x6]
#则:output：[y1,y2,y3],[y4,y5,y6]
#y1=x1w1,1+x4w1,1
#y2=x2w1,2+x5w1,2
#...
#y4=x1w2,1+x4w2,1
#y5=x2w2,2+x5w2,2
#...
x = Input(shape=(lenth_of_seq,))
x_embedding = Embedding(vocab_size, embedding_size, input_length=lenth_of_seq)(x)
hidden_layer = Bidirectional(GRU(GRU_cell,return_sequences=True))(x_embedding)
hidden_layer=Permute((2,1))(hidden_layer)
mid = TimeDistributed(Dense(out_length,use_bias=False))(hidden_layer)
mid=Permute((2,1))(mid)
mid = Bidirectional(GRU(GRU_cell,return_sequences=True))(mid)
y=TimeDistributed(Dense(vocab_size,activation='softmax',use_bias=False))(mid)
model = Model(inputs=x, outputs=mid)
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['acc'])
model.summary()
#版本三，全部attention
x = Input(shape=(lenth_of_seq,))
x_embedding = Embedding(vocab_size, 100, input_length=lenth_of_seq)(x)
hidden_layer = GRU(100,return_sequences=True)(x_embedding)
mid_list=[]
for i in range(out_length):
    mid = Dense(50)(hidden_layer)
    mid = Reshape((1,lenth_of_seq*50))(mid)
    mid_list.append(mid)
mid_2=concatenate(mid_list,axis=1)
y = GRU(100,return_sequences=True)(mid_2)
y=Dense(vocab_size,activation='softmax')(y)
# y=Reshape((out_length,))(y)
model = Model(inputs=x, outputs=y)
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['acc'])
model.summary()
