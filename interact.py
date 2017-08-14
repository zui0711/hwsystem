# coding=utf-8
from PyQt4 import QtCore
import time

import sys
sys.path.append("/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/jwl/PycharmProjects/FailurePredict")
from params import *
from do_seq2seq import *

class BackWorkThread(QtCore.QThread):
    finishSignal = QtCore.pyqtSignal(str)

    def __init__(self, path, mode, parent=None):
        super(BackWorkThread, self).__init__(parent)
        self.mode = mode
        self.path = path
    def run(self):
        time.sleep(3)
        print "PATH IS", self.path
        if self.mode == "train_prepro":
            # prepare_encode_decode_data(prepare_data_source_path,
            #                            prepare_data_source_file,
            #                            prepare_data_save_path,
            #                            prepare_data_label,
            #                            encode_decode_window,
            #                            encode_decode_gap,
            #                            encode_decode_step,
            #                            True,
            #                            2)
            #
            # set_train_test(prepare_data_save_path, prepare_data_label, encode_decode_window, encode_decode_gap,
            #                encode_decode_step)
            #
            # prepare_dialog_data(SAVE_DATA_DIR, seq2seq_vocab_size)

            print "STDOUT TRAIN PREPROCESS..."
        elif self.mode == "train_detect":


            print "STDOUT TRAIN DETECT..."
        elif self.mode == "train_predict":
            mm("t")

            print "STDOUT TRAIN PREDICT..."
        elif self.mode == "run_detect_1":
            print "STDOUT RUN DETECT 1..."
        elif self.mode == "run_detect_2":
            print "STDOUT RUN DETECT 2..."
        elif self.mode == "run_predict":
            print "STDOUT RUN PREDICT..."
        else:
            exit(1)
