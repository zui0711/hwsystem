# coding=utf-8

import sys
from PyQt4 import QtGui, QtCore, uic
from interact import BackWorkThread

from os.path import join as pjoin
import config.path as PATH

qtCreatorFile_main = "uis/main.ui"
qtCreatorFile_train = "uis/train_layout.ui"
qtCreatorFile_param = "uis/param_layout.ui"
qtCreatorFile_run = "uis/run_layout.ui"
qtCreatorFile = "uis/outputWindow.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile_main)
Ui_trainWindow, QttrainClass = uic.loadUiType(qtCreatorFile_train)
Ui_paramWindow, QtparamClass = uic.loadUiType(qtCreatorFile_param)
Ui_runWindow, QtrunClass = uic.loadUiType(qtCreatorFile_run)
Ui_Window, QtClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        # self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.trainButton.clicked.connect(self.print_t)
        self.runButton.clicked.connect(self.print_r)
        PATH.set_path("")

    def print_t(self):
        print("Train...")
        self.train = train()
        self.train.show()
        # self.train.exec_()
        # self.path = self.train.get_path()

    def print_r(self):
        print("Run...")
        self.run = run()
        self.run.show()


class train(QtGui.QMainWindow, Ui_trainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self)
        Ui_trainWindow.__init__(self)
        self.setupUi(self)
        self.loadButton.clicked.connect(self._load_data)
        self.paramButton.clicked.connect(self._set_param)
        self.preproButton.clicked.connect(self._prepro)
        self.detectButton.clicked.connect(self._detect)
        self.predictButton.clicked.connect(self._predict)
        # self.path = path

    def _load_data(self):
        print("Load...")
        print self.lineEdit.text().toUtf8()
        PATH.set_path(str(self.lineEdit.text().toUtf8()))

    def _set_param(self):
        print("set params...")
        set_param = SetParamUi()
        set_param.exec_()

    def _prepro(self):
        print("prepro...")
        ow = outputWindow("train_prepro", u"预处理")
        ow.exec_()

    def _detect(self):
        print("Detect...")
        ow = outputWindow("train_detect", u"故障诊断")
        ow.exec_()

    def _predict(self):
        print('Predict...')
        ow = outputWindow("train_predict", u"故障预测")
        ow.exec_()


class SetParamUi(QtGui.QDialog, Ui_paramWindow):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        Ui_paramWindow.__init__(self)
        self.setupUi(self)
        self.saveButton.clicked.connect(self.save_param)
        self.param_path = pjoin(PATH.get_path(), "params.py")

        self.plainTextEdit.setPlainText(open("config/setting_params.py").read())

    def save_param(self):
        s = self.plainTextEdit.toPlainText()
        open("config/setting_params.py", mode='w').write(s)
        open(self.param_path, mode='w').write(s)


class run(QtGui.QMainWindow, Ui_runWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self)
        Ui_runWindow.__init__(self)
        self.setupUi(self)
        self.lineEdit.setText(PATH.get_path())
        self.loadButton.clicked.connect(self._load_data)
        self.detectButton.clicked.connect(self._detect)
        self.predictButton.clicked.connect(self._predict)

    def _load_data(self):
        print("Load...")
        print self.lineEdit.text().toUtf8()
        PATH.set_path(str(self.lineEdit.text().toUtf8()))

    def _detect(self):
        print("Detect...")
        if self.checkBox.isChecked():
            ow = outputWindow("run_detect_2", u"故障检测")
        else:
            ow = outputWindow("run_detect_1", u"故障检测")
        ow.exec_()
        # detect.exec_()

    def _predict(self):
        print('Predict...')
        ow = outputWindow("run_predict", u"故障预测")
        ow.exec_()


class outputWindow(QtGui.QDialog, Ui_Window):
    def __init__(self, mode, title):
        QtGui.QDialog.__init__(self)
        Ui_Window.__init__(self)
        self.setupUi(self)
        self.setWindowTitle(title)
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

        self.bwThread = BackWorkThread(mode)
        self.bwThread.start()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(str(text))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    # window = Example()
    window.show()
    sys.exit(app.exec_())