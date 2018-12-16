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
with codecs.open(datapaths+"wiki_positive.txt","r","utf-8") as f:
    for line in f:
        text=line.strip().split("\t")[1].replace(" ","")
        texts.append(text)
        
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

# 使用创建模型生成对应 vector

vector=tfidf[corpus[0]]
print(vector)

# 序列化保存

tfidf_corpus=tfidf[corpus]
corpora.MmCorpus.serialize(datapaths+"tfidf_corpus.mm",tfidf_corpus)

# Initializing query structures

index = similarities.MatrixSimilarity(tfidf[corpus])
index.save(datapaths+'tfidf.index')
index = similarities.MatrixSimilarity.load(datapaths+'tfidf.index')

# 进行相似句子查询

sims = index[vector]

# 按照相似度从高到低排序

sims = sorted(enumerate(sims), key=lambda item: -item[1])

# 只输出相似度排名前10的结果
count=0

for sim in sims:
    count += 1
    print(sim[0],sim[1],corpus[sim[0]])

print("#-----------------------------------")
print("\n")

print("#---------------LSI-----------------")
lsi=models.LsiModel(corpus=tfidf_corpus,id2word=dictionary,num_topics=2)
lsi_corpus=lsi[tfidf_corpus]
lsi.save(datapaths+"model.lsi")
corpora.MmCorpus.serialize(datapaths+"lsi_corpus.mm",lsi_corpus)
print("LSI topics")
lsitopics=lsi.print_topics(20)
#print(json.dumps(lsitopics,encoding="utf-8",ensure_ascii=False))
print("#-----------------------------------")
print("\n")

print("#---------------LDA-----------------")
lda=models.LdaModel(corpus=tfidf_corpus,id2word=dictionary,num_topics=2)
lda_corpus=lda[tfidf_corpus]
lda.save(datapaths+"model.lda")
corpora.MmCorpus.serialize(datapaths+"lda_corpus.mm",lda_corpus)
print("LDA Topics:")
ldatopics=lda.print_topics(20)
#print(json.dumps(ldatopics,encoding="utf-8",ensure_ascii=False))
print("#-----------------------------------")
print("\n")
