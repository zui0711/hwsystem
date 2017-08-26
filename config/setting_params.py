# preprocess

encode_decode_window = 50
encode_decode_gap = 2000
encode_decode_step = 40
encode_decode_sample_number = 2

vocab_size = 161



# failure detect

## failure detect

# how to name
MY_W2V_NAME = '50D-w2v.txt' #use this file name when saving word2vec
MY_CNN_MODEL_NAME = 'CNN_detect' #use this file name when saving cnn model
MY_CNN_THRE = 'CNN_threshold' #use this file name when saving cnn threshold
#file quantity
DISCARD_BEGIN = 100
CUT_LINES = 500
SEED = 1
MAX_PER_CLASS = 100 # defines how many files are read
MAXIMUN_FIELS_PER_CLASS_TR = 10000 # train
MAXIMUN_FIELS_PER_CLASS_TE = 3000 #Test
MAX_WORD = 200 # defines how many words are utilized
#w2v
WORD2VEC_SIZE = 50
WORD2VEC_ITER = 2
WORD2VEC_WIN = 5
WORD2VEC_MIN_COUNT = 2
DOC2VEC_SIZE = 100
MY_W2V_DIMENSION = WORD2VEC_SIZE
#cnn
MY_BATCH_SIZE  = 64
MY_EPOC_NUM = 2
MAX_SEQUENCE_LENGTH = 5000
MAX_NB_WORDS = 200
EMBEDDING_DIM = MY_W2V_DIMENSION
VALIDATION_SPLIT = 0.2
#error info
File_label = ['UeAbnormal', 'NORMAL', 'Paging', 'GTPC'] #label info
File_name = [
    'BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag.log',
    "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1.log",
    "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1.log",
    "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1.log",
    ]   #Which files are supposed to be preocessed (infact,this one is unused)
#B
File_name0 = [
    'BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag',
    "BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
    "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1",
    "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1"
    ] #Which files are supposed to be preocessed

#

IF_CUT = False
IF_TRAIN_W2V = False


# failure predict

LSTM_embedding_size = 200
LSTM_batch_size = 128



learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
train_batch_size = 32
predict_batch_size = 128

hidden_size = 64
embedding_size = 128
num_layers = 1
max_train_data_size = 0

steps_per_checkpoint = 100
steps_per_predictpoint = 100

seq2seq_epoch = 100
lstm_epoch = 2
