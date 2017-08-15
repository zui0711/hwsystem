from setting_params import *
import path as PATH

# preprocess
seq2seq_vocab_size = vocab_size + 3

# failure detect



# (self-learning)



# failure predict



LSTM_vocab_size = vocab_size
LSTM_max_len = int(encode_decode_window * 7.2)

BUCKETS = [(LSTM_vocab_size, LSTM_vocab_size)]