# preprocess

encode_decode_window = 50
encode_decode_gap = 2000
encode_decode_step = 10
encode_decode_sample_number = 20

vocab_size = 161



# failure detect



# (self-learning)



# failure predict

LSTM_embedding_size = 200
LSTM_batch_size = 128

prepare_data_source_path = "/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/jwl/PycharmProjects/FailureAnalysis/data/network_diagnosis_data"
prepare_data_source_file = "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"
prepare_data_label = "Paging"
prepare_data_save_path = "/media/zui/work/NETWORK/ALL_DATA/FailurePredict"

learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
train_batch_size = 32
predict_batch_size = 128

hidden_size = 64
embedding_size = 128
num_layers = 1
max_train_data_size = 0

