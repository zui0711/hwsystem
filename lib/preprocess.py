from data_utils import *
from failure_predict import *
from fault_detect import *
from lib.fault_detect_preprocess import *
import sys
sys.path.append("../")

from config.all_params import *


def predict_prepro():
    prepare_encode_decode_data(prepare_data_source_path,
                               prepare_data_save_path,
                               encode_decode_window,
                               encode_decode_gap,
                               encode_decode_step,
                               True,
                               encode_decode_sample_number)


    set_train_test(prepare_data_save_path, encode_decode_window, encode_decode_gap, encode_decode_step)

    prepare_dialog_data(SAVE_DATA_DIR, seq2seq_vocab_size)


def train_prepro():
    print("PROPRESS START\n")
    detect_prepro()
    # predict_prepro()
    print("\nPROPRESS FINISH")



def train_detect():
    print("TRAINING START\n")
    my_fault_detection_train()
    print("\nTRAINING FINISH")


def train_predict():
    print("TRAINING START\n")
    print("     TRAINING seq2seq model\n")
    seq2seq_train()

    print("\n     TRAINING lstm model\n")
    lstm_train()
    print("\nTRAINING FINISH")


def run_detect(mode):
    print("DETECT START\n")
    # print("mode = %d"%mode)
    my_fault_detection_run(mode)
    print("\nDETECT FINISH")


def run_predict():
    print("PREDICT START\n")
    print("     PREDICT seq2seq model\n")
    seq2seq_predict()

    print("\n     PREDICT lstm model\n")
    lstm_predict()
    print("\nPREDICT FINISH")

