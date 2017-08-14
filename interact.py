# coding=utf-8
from PyQt4 import QtCore
import time

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

            print "STDOUT TRAIN PREPROCESS..."
        elif self.mode == "train_detect":
            print "STDOUT TRAIN DETECT..."
        elif self.mode == "train_predict":
            print "STDOUT TRAIN PREDICT..."
        elif self.mode == "run_detect_1":
            print "STDOUT RUN DETECT 1..."
        elif self.mode == "run_detect_2":
            print "STDOUT RUN DETECT 2..."
        elif self.mode == "run_predict":
            print "STDOUT RUN PREDICT..."
        else:
            exit(1)
