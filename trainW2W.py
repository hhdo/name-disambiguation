## SAVE all text in the datasets

import codecs
import json
from os.path import join
import pickle
import os
import re
from utils import *
from tqdm import tqdm

# 增加了v1数据
# pubs_raw = load_json("train","train_pub.json")
# pubs_raw1 = load_json("sna_data","sna_valid_pub.json")
# #pubs_raw2 = load_json("sna_test_data","test_pub_sna.json")

pubs_raw = load_json("all_train_text","all_train_pub.json")
pubs_raw1 = load_json("all_train_text","all_valid_pub.json")

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

f1 = open ('gene/all_text.txt','w',encoding = 'utf-8')

print('reading train pub text:')
for i,pid in enumerate(tqdm(pubs_raw)):
    pub = pubs_raw[pid]
    
    for author in pub["authors"]:
        if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(r,' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f1.write(pstr+'\n')
            
    title = pub["title"]
    pstr=title.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    keyword=""
    if "keywords" in pub:
        for word in pub["keywords"]:
            keyword=keyword+word+" "
    pstr=keyword.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = pub["abstract"]
        pstr=abstract.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        f1.write(pstr+'\n')
        
    venue = pub["venue"]
    pstr=venue.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')

print('reading valid pub text:')
for i,pid in enumerate(tqdm(pubs_raw1)):
    pub = pubs_raw1[pid]
    
    for author in pub["authors"]:
        if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(r,' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f1.write(pstr+'\n')
            
    title = pub["title"]
    pstr=title.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    keyword=""
    if "keywords" in pub:
        for word in pub["keywords"]:
            keyword=keyword+word+" "
    pstr=keyword.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = pub["abstract"]
        pstr=abstract.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        f1.write(pstr+'\n')
        
    venue = pub["venue"]
    pstr=venue.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')

'''
for i,pid in enumerate(pubs_raw2):
    pub = pubs_raw2[pid]
    
    for author in pub["authors"]:
        if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(r,' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f1.write(pstr+'\n')
            
    title = pub["title"]
    pstr=title.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = pub["abstract"]
        pstr=abstract.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        f1.write(pstr+'\n')
        
    venue = pub["venue"]
    pstr=venue.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
'''        
        
f1.close()



from gensim.models import word2vec
print('word2vec training...')
sentences = word2vec.Text8Corpus(r'gene/all_text.txt')
model = word2vec.Word2Vec(sentences, size=200,negative =10, min_count=2, window=4, iter=5,sg=1, hs=1, workers=50)
model.save('word2vec/Aword2vec.model')