from setting_params import *
import path as PATH
from os.path import join as pjoin
import os
import tensorflow as tf
# preprocess
# PATH.set_path("/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/jwl/PycharmProjects/FailureAnalysis/data/network_diagnosis_data")

seq2seq_vocab_size = vocab_size + 3
prepare_data_source_path = PATH.get_path()

prepare_data_source_file = "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1"
# prepare_data_label = "Paging"

prepare_data_save_path = pjoin(PATH.get_path(), "PREDICT")


# SAVE_DATA_DIR = pjoin("/media/workerv/Seagate Backup Plus Drive/ALL_DATA/FailurePredict", "_".join([str(encode_decode_window), str(encode_decode_gap), str(encode_decode_step)]))
SAVE_DATA_DIR = pjoin(prepare_data_save_path, "_".join([str(encode_decode_window), str(encode_decode_gap), str(encode_decode_step)]))
TEST_DATASET_PATH = pjoin(SAVE_DATA_DIR, "test", ".".join(["encode", "ids"+str(seq2seq_vocab_size), "txt"]))

RAW_PAGING_DATA = pjoin(PATH.get_path(), "detect-new/clean/BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1-clean.txt")

# failure detect
#----------------------------------------------------------------------------------------------#
## failure detect

#path
MY_ROOT_PATH = PATH.get_path()
MY_DETECT_ROOT_PATH = os.path.join(MY_ROOT_PATH, 'detect-new')
MY_DATA_PATH = os.path.join(MY_DETECT_ROOT_PATH, 'data')
MY_TRAIN_DATA_PATH = os.path.join(MY_DATA_PATH,'train')
MY_TEST_DATA_PATH = os.path.join(MY_DATA_PATH,'test')
CUT_PATH = os.path.join(MY_DETECT_ROOT_PATH, 'cut')  # for cut
CLEAN_PATH = os.path.join(MY_DETECT_ROOT_PATH, 'clean')  # for clean

FORMATLETTER = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'

MY_ERRORNAME = [
    "USER_CONGESTION",
    "GTPC_TUNNEL_PATH_BROKEN",
    "PROCESS_CPU",
    "SYSTEM_FLOW_CTRL",
    "EPU_PORT_CONGESTION",
    "USER_CONGESTION_RECOVERY",
    "GTPC_TUNNEL_PATH_RECOVERY",
    "PROCESS_CPU_RECOVERY",
    "SYSTEM_FLOW_CTRL_RECOVERY",
    "EPU_PORT_CONGESTION_RECOVERY"
    ]
##-------------------------------------------------------------------------

# failure predict


ERRORNAME = ["USER_CONGESTION",
             "GTPC_TUNNEL_PATH_BROKEN",
             "PROCESS_CPU",
             "SYSTEM_FLOW_CTRL",
             "EPU_PORT_CONGESTION"]

ERRORRECOVERY = ["USER_CONGESTION_RECOVERY",
                 "GTPC_TUNNEL_PATH_RECOVERY",
                 "PROCESS_CPU_RECOVERY",
                 "SYSTEM_FLOW_CTRL_RECOVERY",
                 "EPU_PORT_CONGESTION_RECOVERY"]


tf.app.flags.DEFINE_string('data_dir', SAVE_DATA_DIR, 'data directory')
tf.app.flags.DEFINE_string('model_dir', SAVE_DATA_DIR + '/nn_models', 'Train directory')
tf.app.flags.DEFINE_string('results_dir', SAVE_DATA_DIR + '/results', 'Train directory')

tf.app.flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', learning_rate_decay_factor, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', max_gradient_norm, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('train_batch_size', train_batch_size, 'Batch size to use during training.')
tf.app.flags.DEFINE_integer('predict_batch_size', predict_batch_size, 'Batch size to use during predicting.')

tf.app.flags.DEFINE_integer('vocab_size', seq2seq_vocab_size, 'Dialog vocabulary size.')
tf.app.flags.DEFINE_integer('hidden_size', hidden_size, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('embedding_size', embedding_size, 'Size of embedding.')
tf.app.flags.DEFINE_integer('num_layers', num_layers, 'Number of layers in the model.')

tf.app.flags.DEFINE_integer('max_train_data_size', max_train_data_size, 'Limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', steps_per_checkpoint, 'How many training steps to do per checkpoint.')
tf.app.flags.DEFINE_integer('steps_per_predictpoint', steps_per_predictpoint, 'How many training steps to do per predictpoint.')

FLAGS = tf.app.flags.FLAGS


LSTM_vocab_size = vocab_size
LSTM_max_len = int(encode_decode_window / encode_decode_sample_number * 6)

BUCKETS = [(LSTM_max_len, LSTM_max_len)]

