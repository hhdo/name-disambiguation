# 这个方法在train valid集上效果不好，但是方法的逻辑性较好
import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from utils import *
from tqdm import tqdm
from collections import Counter 
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
    rw_num = 10
    cp=set()
    for k in range(rw_num):
        mpg.generate_WMRW("gene/RW.txt",3,30) #生成路径集
        sentences = word2vec.Text8Corpus(r'gene/RW.txt')
        model = word2vec.Word2Vec(sentences, size=100,negative =20, min_count=1, window=10, workers=50)
        embs=[]
        for i,pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)
                embs.append(np.zeros(100))
        all_embs.append(embs)
    all_embs= np.array(all_embs)
    print ('relational outlier:',len(cp),end = ", ")
    ############################################################### 

    
    
    ##论文文本表征向量
    ###############################################################   
    ptext_emb=load_data('gene','ptext_emb.pkl')
    tcp=load_data('gene','tcp.pkl')
    print ('semantic outlier:',len(tcp))
    tembs=[]
    for pid in pubs:
        tembs.append(ptext_emb[pid])
    ############################################################### 
    


    ##SCI-BERT表征向量
    ###############################################################   
    ptext_emb_scibert=load_data('gene/scibert','paper_embeddings_train_last4321.pkl')
    
    tembs_scibert=[]
    for pid in pubs:
        tembs_scibert.append(ptext_emb_scibert[pid][768*1:768*2])
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
    
    sim = (1.3*np.array(sk_sim) + 1.1*w*np.array(t_sim) + w*np.array(bert_sim))/(1+w+w)
    
    

    ##evaluate
    ###############################################################
    pre = DBSCAN(eps = 0.15, min_samples = 3,metric ="precomputed",n_jobs=-1).fit_predict(sim)
    # 返回每个文章的类标签
    for i in range(len(pre)):
        if pre[i]==-1:
            outlier.add(i)
    print('befer outlier assign:',len(outlier),'/',len(pre),end = ", ")
    
    ## assign each outlier a label
    paper_pair = generate_pair(pubs,outlier)
    
    K = len(set(pre))

    # 建立一个聚类跟随字典
    simDict = dict()
    for aOutlier in outlier:
        maxSimItem = np.argmax(paper_pair[aOutlier])
        if paper_pair[aOutlier][maxSimItem] < 1.5:
            if pre[aOutlier] == -1:
                pre[aOutlier] = K
                K += 1
            continue
        if maxSimItem not in simDict:
            simDict[maxSimItem] = list()
        simDict[maxSimItem].append(aOutlier)

    # 离散聚类生成
    for i_ind in list(simDict.keys()):
        if i_ind in outlier:
            for j_ind in list(simDict.keys()):
                if  i_ind == j_ind:
                    continue
                if i_ind in simDict[j_ind]:
                    simDict[j_ind] += simDict[i_ind]
                    # if j_ind in simDict[j_ind]:
                    #     simDict[j_ind].remove(j_ind)
                    simDict.pop(i_ind)
                    break
    for key in simDict:
        if key in simDict[key]:
            simDict[key].remove(key)

    # 离散点指定到原有簇或新簇
    for p_ind in simDict:
        # 若该点不是离散点，可改为（若该点pre!=-1）
        if p_ind not in outlier:
            for i in simDict[p_ind]:
                pre[i] = pre[p_ind]
#         # 若该点pre!=-1
#         if pre[p_ind] !=-1:
#             for i in simDict[p_ind]:
#                 pre[i] = pre[p_ind]    
        else:
            # 此时p_ind的最佳匹配一定是K簇外的点，证明略。
            # 因此要找到一个K簇内的点抱大腿
            paper_pair1 = paper_pair.copy()
            # 1. p_ind 的次要匹配
            t_leg = np.argmax(paper_pair[p_ind])
            while pre[t_leg]==-1:
                paper_pair[p_ind][t_leg]=-1
                t_leg = np.argmax(paper_pair[p_ind])
            leg1, leg1_w = t_leg, paper_pair[p_ind][t_leg]
            # 2. p_ind 最佳匹配的次要匹配
            # t_leg为p_ind的最佳匹配
            t_leg = np.argmax(paper_pair1[p_ind])
            # t_leg_sec为p_ind最佳匹配的次要匹配
            t_leg_sec= np.argmax(paper_pair1[t_leg])
            while pre[t_leg_sec]==-1:
                paper_pair1[t_leg][t_leg_sec]=-1
                t_leg_sec= np.argmax(paper_pair1[t_leg])
            leg2, leg2_w = t_leg_sec, paper_pair1[t_leg][t_leg_sec]

            # 比较leg1和leg2的权重，取权重最大的那个和阈值比较
            if leg1_w > leg2_w:
                leg, leg_w = leg1, leg1_w
            else:
                leg, leg_w = leg2, leg2_w
            # leg与阈值判断
            if leg_w >= 1.5:
                for i in simDict[p_ind]:
                    pre[i] = pre[leg]
            else:
                for i in simDict[p_ind]:
                    pre[i] = K
                K += 1

    # ## find nodes in outlier is the same label or not
    # for ii,i in enumerate(outlier):
    #     for jj,j in enumerate(outlier):
    #         if jj<=ii:
    #             continue
    #         else:
    #             if paper_pair1[i][j]>=1.5:
    #                 pre[j]=pre[i]
           
    print('after outlier assign:',sum(np.array(list(Counter(pre).values()))==1),'/',len(pre))

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