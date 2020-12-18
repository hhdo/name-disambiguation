import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from utils import *
from tqdm import tqdm
pubs_raw = load_json("train","train_pub.json")
name_pubs = load_json("train","train_author.json")

result=[]
for n,name in enumerate(tqdm(name_pubs)):
    # 遍历每一个重名作者
    ilabel=0
    pubs=[] # all papers
    labels=[] # ground truth
    
    # {
    #     authorID1 : [pubid1,pubid2,...],
    #     authorID2 : [pubid2,pubid6,...],
    #     ...
    # }
    for author in name_pubs[name]:
        iauthor_pubs = name_pubs[name][author]
        for pub in iauthor_pubs:
            pubs.append(pub)
            labels.append(ilabel)
        ilabel += 1
    # pubs存储了当前名字下所有的论文
    # labels存储了pubs中论文对应真实作者的label
    # print (n,name,len(pubs))
    
    
    if len(pubs)==0:
        result.append(0)
        continue
    
    ##保存关系
    ###############################################################
    name_pubs_raw = {}
    # pubs存储了当前名字下所有的论文
    for i,pid in enumerate(pubs):
        name_pubs_raw[pid] = pubs_raw[pid]
        # name_pubs_raw={pid:pid_detail}
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
    rw_num =10
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
    ptext_emb=load_data('gene/scibert','paper_embeddings_text1_1234.pkl')
    tcp=load_data('gene','tcp.pkl')
    # print ('semantic outlier:',tcp)
    tembs=[]
    for i,pid in enumerate(pubs):
        tembs.append(ptext_emb[pid][768*0:768*1])
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
        sk_sim = sk_sim + pairwise_distances(all_embs[k],metric="cosine")
    sk_sim =sk_sim/rw_num 

    
    ##文本相似度
    t_sim = pairwise_distances(tembs,metric="cosine")
    
    # 加权求整体相似度
    w=0.5
    sim = (np.array(sk_sim) + w*np.array(t_sim))/(1+w)
    
    
    
    ##evaluate
    ###############################################################
    pre = DBSCAN(eps = 0.15, min_samples = 3,metric ="precomputed",n_jobs=-1).fit_predict(sim)
    # 返回每个文章的类标签
    
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
            
            
    # 真实标签
    labels = np.array(labels)
    # 预测标签
    pre = np.array(pre)
    # print (labels,len(set(labels)))
    # print (pre,len(set(pre)))
    # 计算p r f1值
    pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(labels,pre)
    print (pairwise_precision, pairwise_recall, pairwise_f1)
    result.append(pairwise_f1)

    print ('avg_f1:', np.mean(result))