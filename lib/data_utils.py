# coding=utf-8

import os
import re
import sys

import numpy as np
np.random.seed(1337)

import random

from os.path import join as pjoin
from config.all_params import *

from tensorflow.python.platform import gfile

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# 正则表达式
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}") # 匹配数字3次+

def b_basic_tokenizer(sentence):
    """Very very basic tokenizer: split the sentence into a list of tokens."""
    words = [w for w in sentence.strip().split() if w]
    return words


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path_list,
                      max_vocabulary_size, tokenizer=None, normalize_digits=True):

    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data " % (vocabulary_path), data_path_list)
        vocab = {}
        counter = 0
        for data_path in data_path_list:
            with gfile.GFile(data_path, mode="r") as f:
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else b_basic_tokenizer(line)
                    for word in tokens:
                        # word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
        #
        # for v in vocab:
        #     print(v, vocab[v], "\n")

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
    vocabulary_path: path to the file containing the vocabulary.

    Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
    ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []

        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
    a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
    if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    for i, ff in enumerate(data_path):
        if not gfile.Exists(target_path[i]):
            print("Tokenizing data in %s" % ff)
            vocab, _ = initialize_vocabulary(vocabulary_path)
            with gfile.GFile(ff, mode="r") as data_file:
                with gfile.GFile(target_path[i], mode="w") as tokens_file:
                    counter = 0
                    for line in data_file:
                        counter += 1
                        if counter % 100000 == 0:
                            print("  tokenizing line %d" % counter)
                        token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")



def prepare_dialog_data(data_dir, vocabulary_size):
    """Get dialog data into data_dir, create vocabularies and tokenize data.

    Args:
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the English vocabulary to create and use.

    Returns:
    A tuple of 3 elements:
      (1) path to the token-ids for chat training data-set,
      (2) path to the token-ids for chat development data-set,
      (3) path to the chat vocabulary file
    """
    # Get dialog data to the specified directory.
    train_path = [pjoin(data_dir, "train", "encode"), pjoin(data_dir, "train", "decode")]
    test_path = [pjoin(data_dir, "test", "encode"), pjoin(data_dir, "test", "decode")]

    train_path_in = [t + ".txt" for t in train_path]
    test_path_in = [t + ".txt" for t in test_path]

    # Create vocabularies of the appropriate sizes.
    vocab_path = pjoin(data_dir, "vocab%d.txt" % vocabulary_size)
    # create_vocabulary(vocab_path, [pjoin(data_dir, "encode.txt"), pjoin(data_dir, "decode.txt")], vocabulary_size)
    create_vocabulary(vocab_path, train_path_in + test_path_in, vocabulary_size)

    # Create token ids for the training data.
    train_ids_path = [t + (".ids%d.txt" % vocabulary_size) for t in train_path]
    data_to_token_ids(train_path_in, train_ids_path, vocab_path)

    # Create token ids for the development data.
    test_ids_path = [t + (".ids%d.txt" % vocabulary_size) for t in test_path]
    data_to_token_ids(test_path_in, test_ids_path, vocab_path)

    return (train_ids_path, test_ids_path, vocab_path)


def read_data(tokenized_dialog_path, max_size=None):
    """Read data from source file and put into buckets.
        Args:
    source_path: path to the files with token-ids.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in BUCKETS]

    with gfile.GFile(tokenized_dialog_path[0], mode="rb") as f_en,\
            gfile.GFile(tokenized_dialog_path[1], mode="rb") as f_de:
        source, target = f_en.readline(), f_de.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                # sys.stdout.flush()

            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(EOS_ID)

            for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            source, target = f_en.readline(), f_de.readline()
    return data_set


def prepare_encode_decode_data(source_path, source_name, save_path,
                               encode_decode_window, encode_decode_gap, encode_decode_step,
                               if_sample=False, sample_number=0):
    """
    为预测模型准备<编码, 解码, 标签>样本
    Args:
        source_path: 原始文件输入路径
        source_name: 原始文件名
        save_path: 保存文件路径
        label: 错误/正常类型
        encode_decode_window: 编解码对窗口大小
        encode_decode_gap: 编解码对间距
        encode_decode_step: 编解码对窗口每次滑动步长
        if_sample: 是否采样取
        sample_number： 采样时多少行取一次

    Returns:

    """
    # 先判断是否已经将原始样本切分为normal/error/recovery三部分
    if not gfile.Exists(pjoin(save_path, "error")):
        print("cut data to normal/error/recovery...")
        cut_data(save_path)

    print("get <encode, decode, label> data...")

    if_err = []
    count = 0
    this_save_path = pjoin(save_path, "_".join([str(encode_decode_window), str(encode_decode_gap), str(encode_decode_step)]))
    gfile.MakeDirs(pjoin(this_save_path, "train"))
    gfile.MakeDirs(pjoin(this_save_path, "test"))
    print("this_save_path = ", this_save_path)
    f_en = gfile.GFile(pjoin(this_save_path, "encode.txt"), mode="wb")
    f_de = gfile.GFile(pjoin(this_save_path, "decode.txt"), mode="wb")
    for num in range(1, len(gfile.ListDirectory(pjoin(save_path, "error")))):
        with gfile.GFile(pjoin(save_path, "normal", str(num)+".txt"), mode="rb") as normal_f, \
             gfile.GFile(pjoin(save_path, "error", str(num)+".txt"), mode="rb") as error_f:
            normal_c = normal_f.readlines()
            error_c = error_f.readlines()
            contxt = normal_c + error_c

            # print("len(contxt) = %d"%len(contxt), "len(normal_c) = %d"%len(normal_c), "len(error_c) = %d"%len(error_c))
            encode_s = 0
            encode_e = encode_s + encode_decode_window

            decode_s = encode_e + encode_decode_gap
            decode_e = decode_s + encode_decode_window

            while (decode_e < len(normal_c)):
                if not if_sample:
                    this_line = " ".join([s.strip() for s in contxt[encode_s: encode_e]])
                    f_en.write(this_line + "\n")
                    this_line = " ".join([s.strip() for s in contxt[decode_s: decode_e]])
                    f_de.write(this_line + "\n")

                else:
                    this_line = " ".join([s.strip() for s in contxt[encode_s: encode_e: sample_number]])
                    f_en.write(this_line + "\n")
                    this_line = " ".join([s.strip() for s in contxt[decode_s: decode_e: sample_number]])
                    f_de.write(this_line + "\n")

                if_err.append("Normal")
                encode_s += encode_decode_step
                encode_e = encode_s + encode_decode_window

                decode_s = encode_e + encode_decode_gap
                decode_e = decode_s + encode_decode_window

            while (encode_e < len(normal_c) and decode_e < len(contxt)):
                if not if_sample:
                    this_line = " ".join([s.strip() for s in contxt[encode_s: encode_e]])
                else:
                    this_line = " ".join([s.strip() for s in contxt[encode_s: encode_e: sample_number]])
                f_en.write(this_line + "\n")

                this_line = ""
                if not if_sample:
                    for line in contxt[decode_s: decode_e]:
                        arr = line.split()
                        write = True
                        for word in ERRORNAME:
                            if word in arr:
                                # print(line, "ERRORNAME")
                                write = False
                                break
                        if write:
                            this_line = " ".join([this_line, line.strip()])
                else:
                    for line in contxt[decode_s: decode_e: sample_number]:
                        arr = line.split()
                        write = True
                        for word in ERRORNAME:
                            if word in arr:
                                # print(line, "ERRORNAME")
                                write = False
                                break
                        if write:
                            this_line = " ".join([this_line, line.strip()])

                f_de.write(this_line + "\n")
                        # outf.write(line.strip()+" SEN_END\n")
                        # outf.write(line.strip() + "\n")

                if_err.append("Error")
                count += 1

                encode_s += encode_decode_step
                encode_e = encode_s + encode_decode_window

                decode_s = encode_e + encode_decode_gap
                decode_e = decode_s + encode_decode_window

            #print("len(if_err)", len(if_err))
            with gfile.GFile(pjoin(this_save_path, "labels.txt"), "wb") as wf:
                for state in if_err:
                    wf.write(state + "\n")


def cut_data(save_path):
    """将原始样本切分为normal/error/recovery三部分"""

    for name in ["error", "normal", "recovery"]:
        gfile.MakeDirs(pjoin(save_path, name))

    ERR_flag = 0
    REC_flag = 0

    cut_start = []  # 故障开始
    cut_mid = []  # 故障结束,开始恢复
    cut_end = [-1]  # 恢复结束

    # 判断每句是否有错误
    # with gfile.GFile(pjoin(source_path, source_name, "clean.txt"), mode="rb") as f:
    with gfile.GFile(RAW_PAGING_DATA, mode="rb") as f:
        contxt = f.readlines()
        print(len(contxt))
        flags = [0 for i in range(len(contxt))] # 0:normal. 1:error, -1:recovery
        for i, line in enumerate(contxt):
            arr = line.split()
            for word in arr:
                if word in ERRORNAME:
                    flags[i] = 1
                    break
                elif word in ERRORRECOVERY:
                    flags[i] = -1
                    break

    # 对error,recovery作切分标记
    write = True
    for i, f in enumerate(flags):
        if f == 1:
            if REC_flag > ERR_flag:
                write = True
                cut_end.append(REC_flag)
            if write:
                cut_start.append(i)
                write = False
            ERR_flag = i
        elif f == -1:
            if REC_flag < ERR_flag:
                cut_mid.append(i)
            REC_flag = i

    for i in range(len(cut_start) - 1):
        with gfile.GFile(pjoin(save_path, "normal", str(i) + ".txt"), mode="wb") as f:
            for line in contxt[cut_end[i] + 1: cut_start[i]]:
                f.write(line)

        with gfile.GFile(pjoin(save_path, "error", str(i) + ".txt"), mode="wb") as f:
            for line in contxt[cut_start[i]: cut_mid[i] - 1]:
                f.write(line)

        with gfile.GFile(pjoin(save_path, "recovery", str(i) + ".txt"), mode="wb") as f:
            for line in contxt[cut_mid[i]:cut_end[i + 1]]:
                f.write(line)


def set_train_test(path, encode_decode_window, encode_decode_gap, encode_decode_step):
    path = pjoin(path, "_".join([str(encode_decode_window), str(encode_decode_gap), str(encode_decode_step)]))
    # with gfile.GFile(pjoin(path, "encode.txt"), mode="rb") as f_en, \
    #         gfile.GFile(pjoin(path, "encode.txt"), mode="rb") as f_de, \
    #         gfile.GFile(pjoin(path, "labels.txt"), mode="rb") as f_l:
    print("set train test...")
    with gfile.GFile(pjoin(path, "encode.txt"), mode="rb") as f_en, \
            gfile.GFile(pjoin(path, "decode.txt"), mode="rb") as f_de, \
            gfile.GFile(pjoin(path, "labels.txt"), mode="rb") as f_l:

        contxt_en = f_en.readlines()
        contxt_de = f_de.readlines()
        contxt_l = f_l.readlines()
        total_len = len(contxt_l)
        indexs = range(total_len)
        if not gfile.Exists(pjoin(path, "train")):
            gfile.MakeDirs((pjoin(path, "train")))
        if not gfile.Exists(pjoin(path, "test")):
            gfile.MakeDirs((pjoin(path, "test")))

        random.shuffle(indexs)

        with gfile.GFile(pjoin(path, "train", "encode.txt"), mode="wb") as f_t_en, \
                gfile.GFile(pjoin(path, "train", "decode.txt"), mode="wb") as f_t_de, \
                gfile.GFile(pjoin(path, "train", "labels.txt"), mode="wb") as f_t_l:
            for i in indexs[ : total_len / 10 * 7]:
                f_t_en.write(contxt_en[i].strip() + "\n")
                f_t_de.write(contxt_de[i].strip() + "\n")
                f_t_l.write(contxt_l[i].strip() + "\n")

        with gfile.GFile(pjoin(path, "test", "encode.txt"), mode="wb") as f_t_en, \
                gfile.GFile(pjoin(path, "test", "decode.txt"), mode="wb") as f_t_de, \
                gfile.GFile(pjoin(path, "test", "labels.txt"), mode="wb") as f_t_l:
            for i in indexs[total_len / 10 * 7 : ]:
                f_t_en.write(contxt_en[i].strip() + "\n")
                f_t_de.write(contxt_de[i].strip() + "\n")
                f_t_l.write(contxt_l[i].strip() + "\n")

# def sampled_


def load_data_lstm(mode):
    name = "decode.ids%d.txt"%seq2seq_vocab_size
    print name

    if mode == "train":
        X_train = []
        y_train = []

        with open(pjoin(SAVE_DATA_DIR, "train", name), "rb") as f,\
                open(pjoin(SAVE_DATA_DIR, "train", "labels.txt"), "rb") as fl:
            con = f.readlines()
            conl = fl.readlines()
            f.close()
            fl.close()

            nor = []
            err = []

            for i, line in enumerate(con):
                if conl[i].strip() == "Normal":
                    nor.append(line)
                else:
                    err.append(line)

            print("train: all - > %d, nor -> %d, err -> %d"%(len(con), len(nor), len(err)))
            for n in nor[:len(err)]:
                X_train.append([int(nn)-2 for nn in n.strip().split()])
                y_train.append(0)
            for e in err:
                X_train.append([int(ee)-2 for ee in e.strip().split()])
                y_train.append(1)

            t = zip(X_train, y_train)
            np.random.shuffle(t)
            for i, elem in enumerate(t):
                X_train[i], y_train[i] = elem
        return_x = X_train
        return_y = y_train


    else:
        X_test = []
        y_test = []

        name = "results(%d, %d).ids%d.%d" % (LSTM_max_len, LSTM_max_len, seq2seq_vocab_size, seq2seq_epoch)

        print pjoin(SAVE_DATA_DIR, "results", name)
        with open(pjoin(SAVE_DATA_DIR, "results", name), "rb") as f, \
                open(pjoin(SAVE_DATA_DIR, "test", "labels.txt"), "rb") as fl:
            con = f.readlines()
            conl = fl.readlines()
            f.close()
            fl.close()

            nor = []
            err = []

            for i, line in enumerate(con):
                if conl[i].strip() == "Normal":
                    nor.append(line)
                else:
                    err.append(line)
            print("test: all - > %d, nor -> %d, err -> %d"%(len(con), len(nor), len(err)))

            for n in nor[:len(err)]:
                X_test.append([int(nn)-2 for nn in n.strip().split()])
                y_test.append(0)
            for e in err:
                X_test.append([int(ee)-2 for ee in e.strip().split()])
                y_test.append(1)

            t = zip(X_test, y_test)
            np.random.shuffle(t)
            for i, elem in enumerate(t):
                X_test[i], y_test[i] = elem

        return_x = X_test
        return_y = y_test

    return return_x, return_y