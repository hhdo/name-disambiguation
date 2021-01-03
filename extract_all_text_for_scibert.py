# extract_all_text.py 用来抽取数据集中所有论文的文本，以训练scibert文本表征embedding
from utils import *
from tqdm import tqdm
import re

pubs_raw_train = load_json("train","train_pub.json")
pubs_raw_valid = load_json("sna_data","sna_valid_pub.json")
pubs_raw_test = load_json("sna_data","sna_test_pub.json")

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

def version1(pubs_raw,filename):
    text_path = 'scibert_pre_data/'+filename
    
    f = open (text_path,'w',encoding = 'utf-8')
    for pid in tqdm(pubs_raw):
        thisPaper = pubs_raw[pid]
        title = thisPaper.get('title','').replace('\n',' ').replace('\\','').strip()
        keywords = ' '.join([i.strip() for i in thisPaper.get('keywords','')]).replace('\n',' ').replace('\\','').strip()
        # venue = thisPaper.get('venue','').strip()
        # year = str(thisPaper.get('year','')).strip()
        # orgs = ' '.join([author.get('org','').strip() for author in thisPaper['authors']])
        abstract = thisPaper.get('abstract','').replace('\n',' ').replace('\\','').strip()
        # [题目 关键词 摘要]
        paperstr = title+' '+keywords+' '+abstract
        paperstr = re.sub(r,' ',paperstr.strip().lower())
        paperstr = re.sub(r'\s{2,}', ' ', paperstr).strip().split()
        if len(' '.join(paperstr))>2:
            paperstr = ' '.join([word for word in paperstr[:512] if len(word)>1]).replace('\t',' ').replace('\n',' ').replace('\\','').strip()
        else:
            paperstr = ' '.join([word for word in paperstr[:512] ]).replace('\t',' ').replace('\n',' ').replace('\\','').strip()
        if len(paperstr) == 0:
            paperstr = 'NONE'
        f.write(pid+'\t'+paperstr+'\n')

    f.close()

def version2(pubs_raw,filename):
    text_path = 'scibert_pre_data/'+filename
    
    f = open (text_path,'w',encoding = 'utf-8')
    for pid in tqdm(pubs_raw):
        thisPaper = pubs_raw[pid]
        title = thisPaper.get('title','').replace('\n',' ').replace('\\','').strip()
        keywords = ' '.join([i.strip() for i in thisPaper.get('keywords','')]).replace('\n',' ').replace('\\','').strip()
        venue = thisPaper.get('venue','').replace('\n',' ').replace('\\','').strip()
        year = str(thisPaper.get('year','')).replace('\n',' ').replace('\\','').strip()
        orgs = ' '.join([author.get('org','').replace('\\','').strip() for author in thisPaper['authors'] if author.get('org','') != ''])
        
        # [关键词 题目 地点 机构 年份]
        paperstr = keywords+' '+title+' '+venue+' '+orgs+' '+year
        paperstr = re.sub(r,' ',paperstr.strip().lower())
        paperstr = re.sub(r'\s{2,}', ' ', paperstr).strip().split()
        if len(' '.join(paperstr))>2:
            paperstr = ' '.join([word for word in paperstr[:512] if len(word)>1]).replace('\t',' ').replace('\n',' ').replace('\\','').strip()
        else:
            paperstr = ' '.join([word for word in paperstr[:512] ]).replace('\t',' ').replace('\n',' ').replace('\\','').strip()
        if len(paperstr) == 0:
            paperstr = 'NONE'
        f.write(pid+'\t'+paperstr+'\n')

    f.close() 


version1(pubs_raw_train,'all_paper_train1.txt')
version1(pubs_raw_valid,'all_paper_valid1.txt')
version1(pubs_raw_test,'all_paper_test1.txt')

# version2(pubs_raw_train,'all_paper_train2.txt')
# version2(pubs_raw_valid,'all_paper_valid2.txt')
# version2(pubs_raw_test,'all_paper_test2.txt')


