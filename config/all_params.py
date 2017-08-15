from setting_params import *
import path as PATH
from os.path import join as pjoin

# preprocess
seq2seq_vocab_size = vocab_size + 3
prepare_data_source_path = "/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/jwl/PycharmProjects/FailureAnalysis/data/network_diagnosis_data"
prepare_data_source_path = PATH.get_path()

prepare_data_source_file = "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"
prepare_data_label = "Paging"
prepare_data_save_path = "/media/zui/work/NETWORK/ALL_DATA/FailurePredict"

# failure detect



# (self-learning)



# failure predict



LSTM_vocab_size = vocab_size
LSTM_max_len = int(encode_decode_window * 7.2)

BUCKETS = [(LSTM_vocab_size, LSTM_vocab_size)]

