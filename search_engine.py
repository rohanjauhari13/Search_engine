#IRWS Part 1
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textmining

text_files=[]
totalDocs=250
# Reading files from system
for i in range(totalDocs):
    text_files.append('C:/Users/admin/Desktop/IRWS\\'+str(i+1)+'.txt')
Doc_array = []
for i in text_files:
    file = open(i, encoding='utf8')
    content = file.read()
    file.seek(0)  
    Doc_array.append(content)

#Step.1 lowering the case of read documents
for i in range(totalDocs):
    Doc_array[i] = Doc_array[i].lower()

#Step.2 removing the punctuation
for i in range(totalDocs):
    remover = str.maketrans('', '', string.punctuation)
    Doc_array[i] = Doc_array[i].translate(remover)

#Step.3 Removing stop words and tockenizing
for i in range(totalDocs):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(Doc_array[i])
    Doc_array[i] = [word for word in word_tokens if word not in stop_words]

#Step.4 De tokenizing 
from nltk.tokenize.treebank import TreebankWordDetokenizer
detoken_doc = []
for i in Doc_array:
    detoken_doc.append(TreebankWordDetokenizer().detokenize(i))


#Step.5 stemming on detockenized terms
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_doc=[]
for doc in detoken_doc:
    stemmed_line=[]
    for term in doc.split():
        stemmed_line.append(ps.stem(term))
    s = ' '.join([str(term) for term in stemmed_line])
    stemmed_doc.append(s)


#Creating term document matrix
tdm = textmining.TermDocumentMatrix()
for i in stemmed_doc:
    tdm.add_doc(i)

dict_tdm = []
for i in tdm.rows(cutoff=1):
    dict_tdm.append(i)


df = pd.DataFrame(dict_tdm)
df.to_csv('Term Document Index.csv', index=True)

#   INVERTED INDEX Creation
#   STEMMING ON TOCKENIZED TERMS

ii = {}
ps=PorterStemmer()
stemmed_array=[]
for sub_doc in Doc_array:
    sub_arr=[]
    for term in sub_doc:
        sub_arr.append(ps.stem(term))
    stemmed_array.append(sub_arr)

for i in range(totalDocs):
    temp = stemmed_array[i]
    for token in temp:
        if token not in ii:
             ii[token] = []
        else:
             ii[token].append(i+1)

inv_ind=ii
s  = pd.Series(ii,index=ii.keys())
df = pd.DataFrame(s)
df.to_csv('Inverted index.csv', index=True)

#PART 2
query=['Argentina- Venezuala agricultural deal food crisis.','Is there some mouse assistive technology for elderly?','Is it Racism proof to play sports? Are Black players safe?','Does DS let players take against 16 people wirelessly. And is it on sale in Europe or Japan ?','Tell me about new marriage rules for people coming from foreign to UK.']

QuerySize=5

#Converting into lower case
for i in range(QuerySize):
    query[i] = query[i].lower()


#Removing punctuation
for i in range(QuerySize):
    translator = str.maketrans('', '', string.punctuation)
    query[i] = query[i].translate(translator)

#Removing Stop Words
for i in range(QuerySize):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(query[i])
    query[i]=[word for word in word_tokens if word not in stop_words]
    
#Stemming the query
stemmed_query=[]
for subquery in query:
    temp=[]
    for term in subquery:
        temp.append(ps.stem(term))
    stemmed_query.append(temp)

#Finding the relevance judgement
relevance_judgement_list=[]
for i in stemmed_query:
    for j in stemmed_array:  
        a_set = set(i)
        b_set = set(j)
        relevance_judgement_list.append(len(a_set & b_set))

relevance_judgement=np.array(relevance_judgement_list).reshape(5,totalDocs)
relevance_judgement_matrix=pd.DataFrame(relevance_judgement)

relevance_judgement_matrix.index=['R1','R2','R3','R4','R5']
col_index=[]
for i in range(totalDocs):
    col_index.append('D'+str(i+1))
relevance_judgement_matrix.columns=col_index
relevance_judgement_matrix.to_csv('Relevance Judgement.csv', index=True)

terms=np.array([])
for i in stemmed_array:
    terms=np.append(terms,np.array(i))
terms.size
unique_terms=np.unique(terms)
unique_terms.size

#term frequency matrix
tf_matrix=np.array([])
for term in unique_terms:
    tf=np.array([])
    for doc in stemmed_array:
        if doc.count(term)!=0:
            tf=np.append(tf,1+np.log10(doc.count(term)))
        else:
            tf=np.append(tf,0)
    tf_matrix=np.append(tf_matrix,tf)
    
#calculating document frequency
df={}
for i in unique_terms:
    df[i]=0
    for doc in stemmed_array:
        if (i in doc)==True:
            df[i]+=1
            
#calculating Inverse document frequency
N=totalDocs
for i in df.keys():
    df[i]=np.log10(N/df[i])
    
#tf-idf matrix
tfidf_matrix=np.array([])
tfidf_doc={}

for term in unique_terms:
    tfidf=np.array([])
    docId=0
    for doc in stemmed_array:
        docId+=1
        if doc.count(term)!=0:
            tfidf_doc[(term,docId)]=1+np.log10(doc.count(term))*df[term]
            tfidf=np.append(tf,(1+np.log10(doc.count(term)))*df[term])
        else:
            tfidf=np.append(tf,0)
            tfidf_doc[(term,docId)]=0
    tfidf_matrix=np.append(tf_matrix,tf)
    
tfidf_query={}
# Query- Does DS technology let players take against people wirelessly. And is it on sale in Europe or Japan ? (245,246)
user_query=stemmed_query[0]
for term in user_query:
    if user_query.count(term)!=0:
        tfidf_query[term]=1+np.log10(user_query.count(term))*df[term]
    else:
        tfidf_query[term]=0
    
cosine_scores={}
for i in range(totalDocs):
    cosine_scores[i+1]=0
for term in user_query:
    if term in inv_ind.keys():
        for docId in inv_ind[term]:           #postings list
            cosine_scores[docId]+=tfidf_query[term]*tfidf_doc[(term,docId)]
    else:
        cosine_scores[docId]=0
for i in range(totalDocs):
    cosine_scores[i+1]/=len(stemmed_array[i])
    
a=sorted(cosine_scores.items(), key = lambda kv:(kv[1], kv[0]))   #sorting on basis of cosine scores of documents.
relevance_score=[ele for ele in reversed(a)]    #collection of relevance documents