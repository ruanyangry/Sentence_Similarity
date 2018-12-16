# _*_ coding:utf-8 _*_

'''
@Author: Ruan Yang
@Date: 2018.12.16
@Purpose: 基于 gensim 使用 TF-IDF,LDA,LSI 进行句子相似性计算
@Reference: https://blog.csdn.net/u013378306/article/details/54633187
'''

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import codecs
import jieba
import json
from gensim import corpora,similarities,models
from collections import defaultdict

# 输入文本预处理

datapaths=r"C:\Users\RY\Desktop\\"

texts=[]
num=0
with codecs.open(datapaths+"wiki_positive.txt","r","utf-8") as f:
    for line in f:
        num += 1
        text=line.strip().split("\t")[1].replace(" ","")
        texts.append(text)
        if num > 5000:
            break
        
raw_texts=texts

# 读取停用词表

stop_words=[]
with codecs.open(datapaths+"stop_words_1893.txt","r","utf-8") as f:
    for line in f:
        stop_words.append(line.strip())

# 对输入文本进行分词操作,去除停用词
# 每一句话对应一个 word list

texts=[[word for word in jieba.cut(text,cut_all=False) if word not in stop_words] for text in texts]

# 去掉只出现一次的单词

frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
        
texts=[[token for token in text if frequency[token]>1] for text in texts]

# 生成/保存词典

dictionary=corpora.Dictionary(texts)
dictionary.save(datapaths+"texts_dict.dic")

# 将文本转化成词袋数据
# 序列化保存文件

corpus=[dictionary.doc2bow(list(text)) for text in texts]
corpora.MmCorpus.serialize(datapaths+"corpus.mm",corpus)

# 首先加载语料库

if os.path.exists("texts_dict.dic") and os.path.exists("corpus.mm"):
    dictionary=corpora.Dictionary.load(datapaths+"texts_dict.dic")
    corpus=corpora.MmCorpus(datapaths+"corpus.mm")
    print("used files generated from string2vector")
else:
    print("Please run string2Vector firstly")
    
print("#-------------TF-IDF-----------------")

tfidf=models.TfidfModel(corpus=corpus)
tfidf.save(datapaths+"model.tfidf")

# Initializing query structures

index = similarities.MatrixSimilarity(tfidf[corpus])
index.save(datapaths+'tfidf.index')
index = similarities.MatrixSimilarity.load(datapaths+'tfidf.index')

# Testing

# 进行相似句子查询

string="鲁迅是中国伟大的文学家"
vec_bow = dictionary.doc2bow([i for i in jieba.cut(string,cut_all=False)])

print("#--------------------------------------#")
print("输入数据转换成词袋形式")
print(vec_bow)
print("#--------------------------------------#")
print("\n")

vector=tfidf[vec_bow]

print("#--------------------------------------#")
print("TF-IDF模型转换得到的vector")
print(vector)
print("#--------------------------------------#")
print("\n")

sims = index[vector]

# 按照相似度从高到低排序

sims = sorted(enumerate(sims), key=lambda item: -item[1])

print("#--------------------------------------#")
print("只输出相似度排名前10的结果")

count=0
for sim in sims:
    count += 1
    print("{} --- {} --- {}".format(sim[0],sim[1],raw_texts[sim[0]]))
    if count > 10:
        break

print("#--------------------------------------#")
print("\n")
