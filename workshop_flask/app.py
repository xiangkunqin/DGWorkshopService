from flask import Flask
from gensim.models import Word2Vec
from gensim import corpora, models, similarities
# en_wiki_word2vec_model = Word2Vec.load('wiki.zh.text.model')
from numpy import vectorize
import numpy as np
from scipy import spatial
import jieba
import jieba.analyse
import jieba.posseg as pseg

import  urllib
import  chardet
import json
app = Flask(__name__)




def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split(' ')
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    ## 取平均值
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)

    # if (n_words > 0):
    #     feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def getSimilar(str1,list_qa,word2vec_model):
    res_list =[]
    a =0
    for item in list_qa:
        a = a+1
        item = item.split(' ')
        if(len(item)==2):
            dict ={}
            q_str = item[0].replace('_','')
            a_str = item[1]

            index2word_set = set(word2vec_model.wv.index2word)
            list_str1 = list(str1)
            _str1 = " ".join(list_str1)
            list_q_str = list(q_str)
            _q_str = " ".join(list_q_str)
            s1_afv = avg_feature_vector(_str1, model=word2vec_model, num_features=100,
                                        index2word_set=index2word_set)
            s2_afv = avg_feature_vector(_q_str, model=word2vec_model, num_features=100,
                                        index2word_set=index2word_set)
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)

            dict["sim"]=sim
            dict["a_str"] = a_str
            dict["q_str"]=item[0]
            print(str(a))
            print(dict)


            if(sim>0):
                res_list.append(dict)
    return res_list

def getListQA(file_path):
    data = []
    for line in open(file_path, "r",encoding="utf8"):  # 设置文件对象并读取每一行文件
        # list_temp=[]
        # list_line = str(line)
        # if(len(list_line)==2):
        #     list_temp.append(list_line)# 将每一行文件加入到list中

        data.append(str(line).replace('\n',''))
    return data

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/getSIMTableName/')
def getSIMTableName():
    word2vec_model = Word2Vec.load('./model/data_zhanyou_1_word2vec.model')
    str1 = "JO MD TFQ"
    list_qa = getListQA("./data/power/data_zhanyou_qa_3.txt")

    # dict_tfidf = getTFIDF(list_qa)
    # print(dict_tfidf)
    # list_qa = getListQA("./data/bank/data_qq_no.txt")
    # str1_cut = " ".join(jieba.cut(str1))

    res = getSimilar(str1, list_qa, word2vec_model)

    if (res != None):
        res.sort(key=lambda k: (k.get('sim', 1)))
        num = 20
        for i in range(len(res)):
            if (num == 0):
                break
            else:
                print(res[len(res) - num])
                num = num - 1
    return str(res)

if __name__ == '__main__':
    app.run()
