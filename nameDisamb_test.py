import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np

pubs_raw = load_json("sna_test_data","test_pub_sna.json")
name_pubs1 = load_json("sna_test_data","example_evaluation_scratch.json")

result={}

for n,name in enumerate(name_pubs1):
    pubs=[]
    for cluster in name_pubs1[name]:
        pubs.extend(cluster)
    
    
    print (n,name,len(pubs))
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
        mpg.generate_WMRW("gene/RW.txt",5,20)
        sentences = word2vec.Text8Corpus(r'gene/RW.txt')
        model = word2vec.Word2Vec(sentences, size=100,negative =25, min_count=1, window=10)
        embs=[]
        for i,pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)
                embs.append(np.zeros(100))
        all_embs.append(embs)
    all_embs= np.array(all_embs)
    print ('relational outlier:',cp)    
    ############################################################### 
 


    ##论文文本表征向量
    ###############################################################  
    ptext_emb=load_data('gene','ptext_emb.pkl')
    tcp=load_data('gene','tcp.pkl')
    print ('semantic outlier:',tcp)
    tembs=[]
    for i,pid in enumerate(pubs):
        tembs.append(ptext_emb[pid])
    ###############################################################
    
    
    
    ##论文相似性矩阵
    ###############################################################
    sk_sim = np.zeros((len(pubs),len(pubs)))
    for k in range(rw_num):
        sk_sim = sk_sim + pairwise_distances(all_embs[k],metric="cosine")
    sk_sim =sk_sim/rw_num    
    

    tembs = pairwise_distances(tembs,metric="cosine")
   
    w=1
    sim = (np.array(sk_sim) + w*np.array(tembs))/(1+w)
    ############################################################### 
    
    
  
    ##evaluate
    ###############################################################
    pre = DBSCAN(eps = 0.2, min_samples = 4,metric ="precomputed").fit_predict(sim)
    pre= np.array(pre)
    
    
    ##离群论文集
    outlier=set()
    for i in range(len(pre)):
        if pre[i]==-1:
            outlier.add(i)
    for i in cp:
        outlier.add(i)
    for i in tcp:
        outlier.add(i)
            
        
    ##基于阈值的相似性匹配
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
    
    for ii,i in enumerate(outlier):
        for jj,j in enumerate(outlier):
            if jj<=ii:
                continue
            else:
                if paper_pair1[i][j]>=1.5:
                    pre[j]=pre[i]
            
    

    print (pre,len(set(pre)))
    
    result[name]=[]
    for i in set(pre):
        oneauthor=[]
        for idx,j in enumerate(pre):
            if i == j:
                oneauthor.append(pubs[idx])
        result[name].append(oneauthor)
    

dump_json(result, "genetest", "result_test.json",indent =4)