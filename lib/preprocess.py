from data_utils import *
from failure_predict import *
from fault_detect import *
from lib.fault_detect_preprocess import *
import sys
sys.path.append("../")

from config.all_params import *


def predict_prepro():
    prepare_encode_decode_data(prepare_data_source_path,
                               prepare_data_source_file,
                               prepare_data_save_path,
                               encode_decode_window,
                               encode_decode_gap,
                               encode_decode_step,
                               True,
                               encode_decode_sample_number)


    set_train_test(prepare_data_save_path, encode_decode_window, encode_decode_gap, encode_decode_step)

    prepare_dialog_data(SAVE_DATA_DIR, seq2seq_vocab_size)


def train_prepro():
    print("train_prepro")
    # detect_prepro()
    predict_prepro()



def train_detect():
    print("train_detect")
    my_fault_detection_train()


def train_predict():
    print("train_predict")
    seq2seq_train()
    lstm_train()


def run_detect(mode):
    print("run_detect", mode)
    my_fault_detection_run(mode)


def run_predict():
    print "run_predict"
    seq2seq_predict()
    lstm_predict()

