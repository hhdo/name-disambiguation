import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from utils_w2w import *
from tqdm import tqdm
pubs_raw = load_json("sna_data","sna_valid_pub.json")
name_pubs1 = load_json("sna_data","sna_valid_example_evaluation_scratch.json")

result={}

for n,name in enumerate(tqdm(name_pubs1)):
    pubs=[]
    for cluster in name_pubs1[name]:
        pubs.extend(cluster)
    
    
    # print (n,name,len(pubs))
    if len(pubs)==0:
        result[name]=[]
        continue
    
    
     ##保存关系
    ###############################################################
    name_pubs_raw = {}
    for i,pid in enumerate(pubs):
        name_pubs_raw[pid] = pubs_raw[pid]
        
    dump_json(name_pubs_raw, 'genename', name+'.json', indent=4)
    save_relation(name+'.json', name)  
    ###############################################################
    
    
    
    ##元路径游走类
    ###############################################################r
    mpg = MetaPathGenerator()
    mpg.read_data("gene")
    ###############################################################
    
  
    
    ##论文关系表征向量
    ############################################################### 
    all_embs=[]
    rw_num = 10
    cp=set()
    for k in range(rw_num):
        mpg.generate_WMRW("gene/RW.txt",3,30) #生成路径集
        sentences = word2vec.Text8Corpus(r'gene/RW.txt')
        model = word2vec.Word2Vec(sentences, size=100,negative =20, min_count=1, window=10)
        embs=[]
        for i,pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)
                embs.append(np.zeros(100))
        all_embs.append(embs)
    all_embs= np.array(all_embs)
    # print ('relational outlier:',cp)
    ############################################################### 

    
    
    ##论文文本表征向量
    ###############################################################   
    ptext_emb=load_data('gene/scibert','paper_embeddings_text1_last4321.pkl')
    tcp=load_data('gene','tcp.pkl')
    # print ('semantic outlier:',tcp)
    tembs=[]
    for i,pid in enumerate(pubs):
        tembs.append(ptext_emb[pid][768*2:768*3])
    ############################################################### 
    
    
    ##SCI-BERT表征向量
    ###############################################################   
    ptext_emb_scibert=load_data('gene/scibert','paper_embeddings_text1_last4321.pkl')
    
    tembs_scibert=[]
    for pid in pubs:
        tembs_scibert.append(ptext_emb_scibert[pid][768*0:768*1])
    ############################################################### 
    
    
    ##离散点
    outlier=set()
    for i in cp:
        outlier.add(i)
    for i in tcp:
        outlier.add(i)
    
    ##网络嵌入向量相似度
    sk_sim = np.zeros((len(pubs),len(pubs)))
    for k in range(rw_num):
        sk_sim = sk_sim + pairwise_distances(all_embs[k],metric="cosine",n_jobs=-1)
    sk_sim =sk_sim/rw_num 

    
    ##文本相似度
    t_sim = pairwise_distances(tembs,metric="cosine",n_jobs=-1)

    #sci-bert相似度
    bert_sim = pairwise_distances(tembs_scibert,metric="cosine",n_jobs=-1)
    
    # 加权求整体相似度
    w=0.5
    
    sim = (1.3*np.array(sk_sim) + w*np.array(t_sim) + w*np.array(bert_sim))/(1+w+w)
    
    
    
    ##evaluate
    ###############################################################
    pre = DBSCAN(eps = 0.15, min_samples = 3,metric ="precomputed",n_jobs=-1).fit_predict(sim)
    print(sum(np.array(pre)==-1),len(pre))
    
    for i in range(len(pre)):
        if pre[i]==-1:
            outlier.add(i)
    
    ## assign each outlier a label
    paper_pair = generate_pair(pubs,outlier)
    paper_pair1 = paper_pair.copy()
    K = len(set(pre))
    for i in range(len(pre)):
        if i not in outlier:
            continue
        j = np.argmax(paper_pair[i])
        while j in outlier:
            paper_pair[i][j]=-1
            j = np.argmax(paper_pair[i])
        if paper_pair[i][j]>=1.5:
            pre[i]=pre[j]
        else:
            pre[i]=K
            K=K+1
    
    ## find nodes in outlier is the same label or not
    for ii,i in enumerate(outlier):
        for jj,j in enumerate(outlier):
            if jj<=ii:
                continue
            else:
                if paper_pair1[i][j]>=1.5:
                    pre[j]=pre[i]
            
    

    # print (pre,len(set(pre)))
    
    result[name]=[]
    for i in set(pre):
        oneauthor=[]
        for idx,j in enumerate(pre):
            if i == j:
                oneauthor.append(pubs[idx])
        result[name].append(oneauthor)
    

dump_json(result, "genetest", "result_valid.json",indent =4)