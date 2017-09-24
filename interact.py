# coding=utf-8
from PyQt4 import QtCore
import time

# import config.path as PATH
from lib.preprocess import *

class BackWorkThread(QtCore.QThread):
    finishSignal = QtCore.pyqtSignal(str)

    def __init__(self, mode, parent=None):
        super(BackWorkThread, self).__init__(parent)
        self.mode = mode

    def __del__(self):
        print("BackWorkThread STOP...")

    def run(self):
        time.sleep(3)
        # print "PATH IS", PATH.get_path()
        if self.mode == "train_prepro":
            train_prepro()
            # print "STDOUT TRAIN PREPROCESS..."

        elif self.mode == "train_detect":
            train_detect()
            # print "STDOUT TRAIN DETECT..."

        elif self.mode == "train_predict":
            train_predict()
            # print "STDOUT TRAIN PREDICT..."

        elif self.mode == "run_detect_1":
            run_detect(0)
            # print "STDOUT RUN DETECT 1..."

        elif self.mode == "run_detect_2":
            run_detect(1)
            # print "STDOUT RUN DETECT 2..."

        elif self.mode == "run_predict":
            run_predict()
            # print "STDOUT RUN PREDICT..."
        else:
            exit(1)
