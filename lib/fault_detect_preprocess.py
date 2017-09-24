# coding=utf-8
from config.all_params import *
from config.setting_params import *
import sys
sys.path.append("../")
import os
import random
import gensim
from multiprocessing import cpu_count
from config.all_params import *
from config.setting_params import *

def format_string(string):
    for c in string:
        if c not in FORMATLETTER:
            string = string.replace(c, ' ')
    retstring = ' '.join(string.split())
    return retstring

def my_de_symbols(file_name, file_label):
    print("Removing symbols from %s"%file_name)
    
    full_file_path = os.path.join(MY_ROOT_PATH,file_name+'.log')
    if not os.path.exists(full_file_path):
        print('%s not found!'%file_name)
        return -1
    dst_file =  os.path.join(CLEAN_PATH,file_name+'-clean.txt')# save to other directory

    cnt = -1
    try:
        f = open(full_file_path, "r")
        wf = open(dst_file, "w+")
        for line in f:
            cnt += 1
            if cnt < DISCARD_BEGIN: continue
            s = format_string(line)
            if s != "":
                wf.write(s + "\n")
          
        wf.close()
        f.close()
    except IOError as err:
        print("Error reading files! Check the file first!")
        print('File error: ' + str(err))
    
    print('Symbols are removed for %s done'%file_name )

def my_train_word2vec_model(file_path,save_path = None, verbose = 1):
    if verbose >= 1:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    class MySentences(object):
        'path-files'
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()
                    
    class MySentences2Hier(object):
        'pathA-pathB-files'
        def __init__(self, dirname):
            self.dirname = dirname
        def __iter__(self):
            first_hier = os.listdir(self.dirname)
            random.shuffle(first_hier)
            for folder_name in first_hier:
                new_folder_path = self.dirname + os.sep + folder_name
                second_hier = os.listdir(new_folder_path)
                random.shuffle(second_hier)
                for fname in second_hier:
                    for line in open(os.path.join(new_folder_path, fname)):
                        yield line.split() # tokenizer: space 

    print('Obtaining word2vec model...')
#     sentences = MySentences2Hier(file_path) # a memory-friendly iterator
    sentences = MySentences(file_path)
    model = gensim.models.Word2Vec(sentences,size=WORD2VEC_SIZE, iter= WORD2VEC_ITER, window=WORD2VEC_WIN, min_count= WORD2VEC_MIN_COUNT , workers=cpu_count())
    if save_path != None:
        save_dir = os.path.join(save_path,'word2vec.model')
        model.save(save_dir,ignore=[])  # m  odel.save('test',ignore=[]) # there's an ignore-setting which defaults to ['syn0norm'] 
        print("Word2vec Model saved successfully.")
    return model 

def my_w2v_to_txt(model_path,output):
    'file_path: trained word2vec model'
    'save word2vec results to text file(default file names)'
    'format:  hello 0 .2 3.1 ... '
    if isinstance(model_path, gensim.models.Word2Vec):
        print('Write word  vectors into text from model instance.')
        model = model_path
    else:
        print('Load model and write word  vectors into text.')
        model = gensim.models.Word2Vec.load(model_path) 
    
    # vocab = list(model.vocab.keys()) # model.vocab  Vocab(count:7, index:11550, sample_int:4294967296L)
    # vd= model.vocab
    all_words =model.wv.index2word
    # prepare a dictionary for these vector:
    word_vec = {}
    for item in all_words:
        if item not in word_vec.keys():
#         if not word_vec.has_key(item):
            word_vec[item] = model[item]  

    output_file = os.path.join(output,MY_W2V_NAME)
    with open(output_file,"w+") as f:
        for d,x in word_vec.items():
            f.write(d+' ')
            tt=''
            xx =  [ str(i) for i in x ]
            tt= ' '.join(xx)
            f.write(tt)
            f.write('\n')
    print('All word vectors have been written to text!')

## cut and split
def my_cut_split_data( file_name, file_label, cut_num=500):

    src_file =  os.path.join(CLEAN_PATH,file_name+'-clean.txt')
    print(CLEAN_PATH)
    if not os.path.exists(src_file):
        print('%s not found!'%file_name)
        return -1
    print('Cutting data...')
    file_count = 0     # counter for file numbers
    line_count = 0   # counter for the lines of a file
    file_empty = True  # flag 
    if_begin = True # if ture, create a new file
    d_train = os.path.join(MY_TRAIN_DATA_PATH,file_label)
    d_test = os.path.join(MY_TEST_DATA_PATH,file_label)
    dst_folder = [d_train,d_test]
    for f in dst_folder:
        if not os.path.exists(f):
            os.makedirs(f)
    contxt = open(src_file, "r")
    for i, line in enumerate(contxt):
        if i % 3 == 0:
            if if_begin:
                if random.random() > 0.7:
                    dst_file = os.path.join(dst_folder[1], str(file_count)+'.txt')
                else:
                    dst_file = os.path.join(dst_folder[0], str(file_count)+'.txt')
                wwf = open(dst_file, "w")
                if_begin = False
            arr = line.split()
            can_write = True

            # judge if some error occurs in this line
            for err_type in ERRORNAME:
                if err_type in arr:
                    can_write = False
                    break
            if can_write:
                line_count += 1
                wwf.write(line)
            # end of a file
            if line_count == cut_num:
                line_count = 0
                file_count += 1
                if_begin = True
                wwf.close()
    
    contxt.close()
    if line_count != cut_num:
        wwf.close()
        os.remove(dst_file)
        
    print('Current file type is %s.'%file_label)
    print('%s files are prepared.'%file_count)

def detect_prepro():
    path_init()
    
    #remove symbos
    if IF_CLEAN:
        for i in range(len(File_name0)):
            my_de_symbols(File_name0[i],File_label[i])

    #train word2vectors
    if  IF_TRAIN_W2V :
        t_model=my_train_word2vec_model(CLEAN_PATH,save_path = MY_DETECT_ROOT_PATH )
        my_w2v_to_txt(t_model,MY_DETECT_ROOT_PATH)
    #cut data
    if   IF_CUT :
        for i in range(len(File_name0)):
            my_cut_split_data(File_name0[i],File_label[i],cut_num=CUT_LINES) #3210
   
    print('End of pre-preocess for detection!')

def path_init():
    'initialize all path'
    if not os.path.exists(MY_DETECT_ROOT_PATH):
        os.makedirs(MY_DETECT_ROOT_PATH)
    
    if not os.path.exists(MY_DATA_PATH):
        os.makedirs(MY_DATA_PATH)
    
    if not os.path.exists(MY_TRAIN_DATA_PATH):
        os.makedirs(MY_TRAIN_DATA_PATH)
    
    if not os.path.exists(MY_TEST_DATA_PATH):
        os.makedirs(MY_TEST_DATA_PATH)   
    
    if not os.path.exists(CUT_PATH):
        os.makedirs(CUT_PATH)
        
    if not os.path.exists(CLEAN_PATH):
        os.makedirs(CLEAN_PATH)