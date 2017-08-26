import sys
sys.path.append("../")
from config.all_params import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM


def create_model_lstm(embedding_trainable=True, embedding_matrix=None):
    model = Sequential()
    if embedding_trainable:
        model.add(Embedding(input_dim=LSTM_vocab_size + 1, output_dim=LSTM_embedding_size))
    else:
        model.add(Embedding(input_dim=LSTM_vocab_size+1, output_dim=LSTM_embedding_size,
                            weights=[embedding_matrix], input_length=LSTM_max_len, trainable=False))

    model.add(LSTM(LSTM_embedding_size, dropout_W=0.1, dropout_U=0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
