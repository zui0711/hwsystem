# coding=utf-8
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append("../")
from config.all_params import *
from config.setting_params import *

from lib.data_utils import *


import numpy as np

import logging,sys,numpy
import json,pickle,os,random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
# from keras.callbacks import ModelCheckpoint

import keras.backend as K



batch_size = MY_BATCH_SIZE 
epoch_num = MY_EPOC_NUM
EMBEDDING_DIM = MY_W2V_DIMENSION
word2vec_results = os.path.join(MY_DETECT_ROOT_PATH, MY_W2V_NAME)
#==========================================================================================
def cnn_w2v_adptive_train():
    K.clear_session()
    N = MAXIMUN_FIELS_PER_CLASS_TR
    'use fixed embedding weights'
    print('Indexing word vectors from:')
    print(word2vec_results)
    embeddings_index = {} # embedding dict
    f = open(word2vec_results,'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    # second, prepare text samples and their labels

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    TEXT_DATA_DIR = MY_TRAIN_DATA_PATH
    print('Processing dataset for training.')
    print('The directory is %s'%TEXT_DATA_DIR)
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path): # if dir
            label_id = len(labels_index)
            labels_index[name] = label_id
            j = 0
            for fname in sorted(os.listdir(path)):
                if j < N: 
                    fpath = os.path.join(path, fname)
                    f = open(fpath, 'r')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)
                    j = j + 1
#             j = 0         
    print('Load  %s samples.' % len(texts))
    print(labels_index)
    print(set(labels))

    labels_index_path = os.path.join(MY_DETECT_ROOT_PATH, 'labels_index')
    with open(labels_index_path, 'wb') as ft:
        pickle.dump(labels_index, ft, protocol=1)

    filterstr = r'!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n'
    tokenizer = Tokenizer(filters=filterstr, nb_words=MAX_NB_WORDS) # tokenize num_words
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index # dict  like hello:23
    print('Found %s unique tokens.' % len(word_index))
    tokenizer_path =  os.path.join(MY_DETECT_ROOT_PATH,'tokenizer') 
    with open(tokenizer_path,'wb') as ft:
        pickle.dump(tokenizer,ft, protocol=1)
    x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #return nparray
    y_train = to_categorical(np.asarray(labels))# to 0 1 vector
    print('Shape of data tensor:', x_train.shape)
    print('Shape of label tensor:', y_train.shape)
    
    print('Preparing embedding matrix.')
    nb_words = min(MAX_NB_WORDS, len(word_index)) #
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))# embedding  mat
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            'no operation'
            continue
        embedding_vector = embeddings_index.get(word) # corresponding word vectors
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector # row -- word
    
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True) # weights-training is disabled You called `set_weights(weights)` on layer "embedding_1" with a  weight list of length 1, but the layer was expecting 0 weights. Provided weights:
    
    print('Training CNN model...')
    
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    class_num = len(np.unique(labels))
    preds = Dense(len(labels_index), activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    print('Fitting ..')
    t_path = os.path.join(MY_DETECT_ROOT_PATH,MY_CNN_MODEL_NAME) 
    filepath = t_path + '_best.h5'
 
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')#save_best_only=True,
    # callbacks_list = [checkpoint] # checkpoint
    # print(VALIDATION_SPLIT,epoch_num,batch_size)
    model.fit(x_train, y_train,  validation_split=VALIDATION_SPLIT,
              nb_epoch=epoch_num, batch_size=batch_size, verbose=1, shuffle=True)#  callbacks=None
    print('Saving model　at %s.'%filepath)
    json_string = model.to_json()  #equals json_string = model.get_config()
    with open(t_path +'.json','w+') as ft:
        json.dump(json_string, ft)

    model.save_weights(t_path+'.h5', overwrite=True) # save weights
    model.save(t_path)  # save all 

    ### calculate threshhold
    print('Calculating threshold...')
    
    p = model.predict(x_train)
    ground_truth = labels
    pred = numpy.argmax(p,axis=1).tolist()
    # pred = p.argmax(axis=-1)
    p_m = numpy.amax(p, axis=1).tolist() #maximum of each row
    p_sep = dict()  # propability for different type
    thre_dict = dict() # 0: mean,std
    y_list = set(ground_truth)
    print(y_list)
    for i in y_list:
        p_sep[i] = []

    for i in range(x_train.shape[0]):
        if  ground_truth[i] == pred[i]:
            p_sep[ ground_truth[i] ].append(p_m[i])
            
    for i in y_list:
        thre_dict[i] = ( numpy.mean(p_sep[i]), numpy.std(p_sep[i]) )
    print(thre_dict)
    ## save threshold
    thre_path = os.path.join(MY_DETECT_ROOT_PATH, MY_CNN_THRE)
    with open(thre_path,'wb') as ft:
        pickle.dump(thre_dict, ft )    
    with open(thre_path + '.raw','wb') as ft:
        pickle.dump(p, ft )
    with open(thre_path+'label.dict','wb') as ft:
        pickle.dump(labels_index,ft)
    print('Threshold saved')
    print('End of this process.')
    
##=================================================================================================
def cnn_w2v_test():
    K.clear_session()
    N = MAXIMUN_FIELS_PER_CLASS_TE
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    TEXT_DATA_DIR = MY_TEST_DATA_PATH
    print('Processing dataset for trainitestingng.')
    print('The directory is %s'%TEXT_DATA_DIR)
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path): #  
            label_id = len(labels_index)
            labels_index[name] = label_id
            j = 0
            for fname in sorted(os.listdir(path)):
                if j < N: 
                    fpath = os.path.join(path, fname)
                    f = open(fpath, 'r')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)
                    j = j + 1
            j = 0         
    print('Load  %s samples.' % len(texts))
    
#     tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # tokenize
    tokenizer_path =  os.path.join(MY_DETECT_ROOT_PATH,'tokenizer') 
    with open(tokenizer_path,'rb') as ft:
        tokenizer = pickle.load(ft)
#     tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index # dict  like hello:23
    print('Found %s unique tokens.' % len(word_index))
    
    x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #return nparray
    y_test = to_categorical(np.asarray(labels))# to 0 1 vector
    print('Shape of data tensor:', x_test.shape)
    print('Shape of label tensor:', y_test.shape)
 
    from keras.models import load_model
    t_path = os.path.join(MY_DETECT_ROOT_PATH,MY_CNN_MODEL_NAME) 
    print('Evaluating...')
    model = load_model(t_path)
    scores = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=1 )#,  batch_size=batch_size
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     
    # # model reconstruction from JSON: these lines also works
    # from keras.models import model_from_json
    # filepath = t_path + '_best.h5'
    # j_path = t_path + '.json'
    # with open(j_path,'r') as ft:
    #     j_string = json.load(ft)
    # model_j = model_from_json(j_string)
    # model_j.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #                metrics=['acc'])
    # model_j.load_weights(filepath)
    # scores = model_j.evaluate(x_test, y_test,  batch_size=batch_size, verbose=1 ) #compile first when evaluate
    # print("\nTesting:%s: %.2f%%" % (model_j.metrics_names[1], scores[1]*100))
      

    print('End of this process.')
#     from keras.utils.visualize_util import plot
#     data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
#     plot(model, to_file=r'.\data\cnn-embedding-model'+data_today+'.png')

##==============================================================================================
## set threshold in advance
# thre_ue = 0.84
# thre_pa = 0.96
# thre_nor = 0.8
# thre_gt = 0.75
##
def cnn_w2v_adaptive_test():
    K.clear_session()
    N = MAXIMUN_FIELS_PER_CLASS_TE
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    TEXT_DATA_DIR = MY_TEST_DATA_PATH
    print('Processing dataset for trainitestingng.')
    print('The directory is %s'%TEXT_DATA_DIR)
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path): #  
            label_id = len(labels_index)
            labels_index[name] = label_id
            j = 0
            for fname in sorted(os.listdir(path)):
                if j < N: 
                    fpath = os.path.join(path, fname)
                    f = open(fpath, 'r')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)
                    j = j + 1

    print('Load  %s samples.' % len(texts))
    print(labels_index)
    # print(set(labels))

    labels_index_path =  os.path.join(MY_DETECT_ROOT_PATH,'labels_index')
    with open(labels_index_path,'rb') as ft:
        old_labels_index = pickle.load(ft)
    if 'NORMAL' in labels_index:
        normal_idx = labels_index['NORMAL']
    else:
        normal_idx = 1
        print("Please revise the KEY here!")
    if not old_labels_index == labels_index:
         print('Something wrong has occured.')
         labels_index_path = os.path.join(MY_DETECT_ROOT_PATH, 'labels_index')
         with open(labels_index_path, 'rb') as ft:
             old_labels_index = pickle.load(ft)
         if 'NORMAL' in labels_index:
             normal_idx = labels_index['NORMAL']
         else:
             normal_idx = 1
             print("Please revise the KEY here!")
         if not old_labels_index == labels_index:
             print('Something wrong has occured.')
             os.system('pause')





#     tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # tokenize
    tokenizer_path =  os.path.join(MY_DETECT_ROOT_PATH,'tokenizer') 
    with open(tokenizer_path,'rb') as ft:
        tokenizer = pickle.load(ft)
#     tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index # dict  like hello:23
    print('Found %s unique tokens.' % len(word_index))
    
    x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #return nparray
    y_test = to_categorical(np.asarray(labels))# to 0 1 vector
    print('Evaluating %s samples...'%x_test.shape[0])
 
    ## adaptive juding
    thre_path = os.path.join(MY_DETECT_ROOT_PATH, MY_CNN_THRE)
    pkl_file = open(thre_path, 'rb')
    thre_dict = pickle.load(pkl_file) #0: mean,std
    thre_new = dict()
    for key,val in  thre_dict.items():
        # 'cal threshold here'
        # print(key)
        # print(val)
        thre_new[key] = val[0] - 2*val[1]

    print('new threshold...')
    print(thre_new)

    from keras.models import load_model
    t_path = os.path.join(MY_DETECT_ROOT_PATH,MY_CNN_MODEL_NAME) 
    print('Loading pre-trained model...')
#     from IPython.core.debugger import Tracer
#     Tracer()() #this one triggers the debugger
    model = load_model(t_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    p = model.predict(x_test,verbose=1)
    # pred = p.argmax(axis=-1)
    pred = numpy.argmax(p,axis=1).tolist()
    ground_truth = labels
    # ground_truth = numpy.amax(y_test, axis=1).tolist() #y_test.argmax(axis=-1)
    p_m = numpy.amax(p, axis=1).tolist() #maximum of each row
    cnt1 = 0 # less then threshold
    cnt2 = 0
    cnt3 = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)  # ,  batch_size=batch_size
    for i in range(p.shape[0]):
        if pred[i] == ground_truth[i]:
            cnt3 += 1

        if p_m[i] < thre_new[ pred[i] ]:
            cnt1 += 1
        elif  pred[i] == ground_truth[i]:
            cnt2 +=1

    for i in range(p.shape[0]):
        if ground_truth[i] == normal_idx:  # positive samples
            if pred[i] == ground_truth[i]:
                TP += 1
            else:
                FN += 1
        if ground_truth[i] != normal_idx:  # error samples
            if pred[i] != normal_idx:
                TN += 1
            else:
                FP += 1
                    #             if pred[i] == ground_truth[i]:
                    #                 TN += 1
                    #             else:
                #                 FP += 1
    P = TP *1./ (TP + FP)
    R = TP *1./ (TP + FN)
    F1 = 2. * TP / (2 * TP + FP + FN)
    print('Accuracy:%s; Recall:%s;F1 score:%s' % (P, R, F1))

    acc = cnt2*100./x_test.shape[0]
    below_thre = cnt1*100./x_test.shape[0]

    print("acc = %.2f%% " %   acc)
    print('%.2f%% samples are unkown.'% below_thre )
    # print('score:',scores[1]) # ,  batch_size=batch_size
    print('raw score:',cnt3*100./x_test.shape[0])
    # print('raw score:',scores[1])
    # # model reconstruction from JSON: these lines also works
#     from keras.models import model_from_json
#     filepath = t_path + '_best.h5'
#     j_path = t_path + '.json'
#     with open(j_path,'r') as ft:
#         j_string = json.load(ft)
#     model_j = model_from_json(j_string)
#     model_j.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['acc'])
#     model_j.load_weights(filepath)
#     scores = model_j.evaluate(x_test, y_test,  batch_size=batch_size, verbose=1 ) #compile first when evaluate
#     print("Testing:%s: %.2f%%" % (model_j.metrics_names[1], scores[1]*100))
    

    print('End of this process.')
#     from keras.utils.visualize_util import plot
#     data_today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
#     plot(model, to_file=r'.\data\cnn-embedding-model'+data_today+'.png')
## =====================================================================================================
def my_fault_detection_run(  if_adaptive = 0):
    K.clear_session()
    'r_path is the root path for all ops'
    if if_adaptive == 1:
        print('Adaptive mode is enabled!')
        cnn_w2v_adaptive_test()
    else:
        print('Adaptive mode is disabled!')
        cnn_w2v_test()
   
def my_fault_detection_train(r_path = MY_ROOT_PATH):
    print('Training procedure for fault detectioin.')
    cnn_w2v_adptive_train()
    
