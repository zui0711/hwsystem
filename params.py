# preprocess

_encode_decode_window = 50
_encode_decode_gap = 2000
_encode_decode_step = 10
_encode_decode_sample_number = 20

_seq2seq_vocab_size = 164



# failure detect



# (self-learning)



# failure predict

_LSTM_vocab_size = _seq2seq_vocab_size - 3
_LSTM_max_len = 360
_LSTM_embedding_size = 200
_LSTM_batch_size = 128

_prepare_data_source_path = "/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/jwl/PycharmProjects/FailureAnalysis/data/network_diagnosis_data"
_prepare_data_source_file = "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"
_prepare_data_label = "Paging"
_prepare_data_save_path = "/media/zui/work/NETWORK/ALL_DATA/FailurePredict"

_learning_rate = 0.5
_learning_rate_decay_factor = 0.99
_max_gradient_norm = 5.0
_train_batch_size = 32
_predict_batch_size = 128

_hidden_size = 64
_embedding_size = 128
_num_layers = 1

_max_train_data_size = 0

_BUCKETS = [(360, 360)]
