#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://github.com/suchartee/topic_modeling


# # Libraries (RUN THIS)

# In[1]:


import pandas as pd
import re
import requests
import string
import time
import sys
from bs4 import BeautifulSoup
# !pip install nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
# !pip install gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2019)
from gensim import corpora, models
from pprint import pprint
# !pip install pyldavis
import pyLDAvis
import pyLDAvis.gensim
import warnings
warnings.filterwarnings('ignore')
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# !pip install xlrd
import xlrd
import os
from nltk.tag import pos_tag
import os
import math
import numpy as np 
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from itertools import chain
import collections
import operator
from operator import itemgetter
import numpy
# !pip install openpyxl
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5


# # Getting Dataset from the URLs (Run this only once)

# In[ ]:


# GET .JSON FILE FROM GITHUB
# read .json to list
with open('News_Category_Dataset_v2.json', 'r') as f:
    data = f.readlines()
# remove the trailing "\n" from each line
data = map(lambda x: x.strip(), data)
data_json_str = "[" + ','.join(data) + "]"
# load it into df
data_df = pd.read_json(data_json_str)


# In[ ]:


### THIS WOULD TAKE 3 DAYS TO GET ALL
# soup content from data_df['urls']
start = time.time()
t = []
c = []
for i, url in enumerate(data_df['link']):
    temp = re.sub(".*https?:\/\/", "http://", url)
    # get only huffpost website
    if 'huffingtonpost' in temp:
        try:
            headers = {'User-Agent': 'Chrome/50.0.2661.102'}
            html = requests.get(temp, headers=headers) 
            soup = BeautifulSoup(html.content, "html5lib")
            text = [p.text for p in soup.find_all("div", {"class": "content-list-component yr-content-list-text text"})]
            t.append(text)
            c.append(data_df['category'][i])
            print(i, temp, ((time.time() - start)/60)/60)
        except Exception:
            pass
print("Got all soup: " + str((time.time() - start)/60) + " mins")


# In[ ]:


# convert from df to xlsx for later uses
# join contents for each url
t = [" ".join(i) for i in t]
# df of transcript and category
df = pd.DataFrame([t, c]).transpose()
# name the columns
df.columns = ['transcript','category']
# remove all the blank content
df = df[df['transcript'] != '']
# export .xlsx
df.to_excel("transcript.xlsx", index=False)


# # Pre clean the dataset (Run this only once)

# In[ ]:


if df[df.isna().any(axis=1)].shape[0] != 0:
    df = df.dropna()


# In[ ]:


def clean_text_huffpost(text):
    # NNP (proper noun)
    tagged_sent = pos_tag(text.split())
    propernouns = [word for word,pos in tagged_sent if pos == 'NNP' or pos == 'NNPS']
    for word in propernouns:
        text =  text.replace(word, ' ')   
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # remove special char
    text = re.sub('[‘’“”…―]|\n|\xa0|\w*\d\w*', ' ', text)
    # remove any single character
    text = re.sub(' +\w{1} +', ' ', text)
    # change extra whitespace to 1
    text = re.sub('[ ]{2,}', ' ', text).strip()
    return text


# In[ ]:


df_2['transcript'] = df_2['transcript'].apply(lambda x: clean_text_huffpost(x))
df_2.to_csv('df_2.csv')
df_2.head()


# # Retrieve cleaned dataframe (before lemma and stem)

# In[2]:


sav_loc = r'C:\Users\Alice\Dropbox (CSU Fullerton)\CSUF\Graduate Project\topic_modeling\topic_modeling\data' # location where df_2.csv is saved
os.chdir(sav_loc)
df_2 = pd.read_csv('df_2.csv')


# In[5]:


df_2.shape


# In[6]:


df_2.category.value_counts()


# # Lemma
# min 15, max 0.5, 10 topics

# In[7]:


# slice data into half
# size = int(df_2.shape[0]/2)
# new_df_2 = df_2[:size]
new_df_2 = df_2
transcript2 = df_2.transcript.to_list()


# In[8]:


# get list of stop words provided by NLTK and Gensim
nltk_stop_words = list(stopwords.words('english'))
gensim_stop_words = list(gensim.parsing.preprocessing.STOPWORDS)
stop_words = list(set(nltk_stop_words) | set(gensim_stop_words))

def lemma_stem(text):
#   stemmer = PorterStemmer()
#   stemmer = SnowballStemmer("english")
#   return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return WordNetLemmatizer().lemmatize(text, pos='n')

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words:
            result.append(lemma_stem(token))
    return result

# lemma and remove stop words
preprocessed_transcript2_2 = new_df_2['transcript'].fillna('').astype(str).map(preprocess)


# In[9]:


preprocessed_transcript2_2.head()


# # Topic Coherence
# Finding the optimal number of topics by topic coherence
# 1. C_v measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
# 2. C_p is based on a sliding window, one-preceding segmentation of the top words and the confirmation measure of Fitelson’s coherence
# 3. C_uci measure is based on a sliding window and the pointwise mutual information (PMI) of all word pairs of the given top words
# 4. C_umass is based on document cooccurrence counts, a one-preceding segmentation and a logarithmic conditional probability as confirmation measure
# 5. C_npmi is an enhanced version of the C_uci coherence using the normalized pointwise mutual information (NPMI)
# 6. C_a is baseed on a context window, a pairwise comparison of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity

# Intrinsic Measure
# It is represented as UMass. It measures to compare a word only to the preceding and succeeding words respectively, so need ordered word set.It uses as pairwise score function which is the empirical conditional log-probability with smoothing count to avoid calculating the logarithm of zero.
# 
# Extrinsic Measure
# It is represented as UCI. In UCI measure, every single word is paired with every other single word. The UCIcoherence uses pointwise mutual information (PMI).
# 
# 
# Both Intrinsic and Extrinsic measure compute the coherence score c (sum of pairwise scores on the words w1, …, wn used to describe the topic)
# 
# - https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
# - https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#14computemodelperplexityandcoherencescore
# 
# - https://www.aclweb.org/anthology/N10-1012.pdf
# 

# ## with bi gram
# 20 passes, 10 topics

# In[10]:


# bigram
bigram = gensim.models.Phrases(preprocessed_transcript2_2, min_count=5, threshold=50) # higher threshold fewer phrases
bigram_list = []
for i in range(len(preprocessed_transcript2_2)):
    bigram_list.append(bigram[preprocessed_transcript2_2[i]])


# In[11]:


# create dictionary of all words (bi gram)
dictionary2_bi = gensim.corpora.Dictionary(bigram_list)
print("Dictionary size before filtering: " + str(len(dictionary2_bi)))

dictionary2_bi.filter_extremes(no_below=15, no_above=0.5, keep_n=50000)
print("Dictionary size after filtering: " + str(len(dictionary2_bi)))

# bag of words
bow_corpus2_bi = [dictionary2_bi.doc2bow(doc) for doc in bigram_list]

# applying tfidf - corresponding to the arguments in filter_extremes()
tfidf2_bi = models.TfidfModel(bow_corpus2_bi)
corpus_tfidf2_bi = tfidf2_bi[bow_corpus2_bi]


# In[12]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    umass_values = []
    uci_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=3)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        umass = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        umass_values.append(umass.get_coherence())
        uci = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_uci')
        uci_values.append(uci.get_coherence())

    return model_list, coherence_values, umass_values, uci_values


# In[13]:


model_list, coherence_values, umass_values, uci_values = compute_coherence_values(dictionary=dictionary2_bi, corpus=corpus_tfidf2_bi, texts=bigram_list, start=2, limit=11, step=1)


# In[14]:


# Show c_v graph
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[15]:


for i in range(len(model_list)):
  print("Model: " + str(model_list[i]))
  print("C_V: " + str(coherence_values[i]))
  print("Umass: " + str(umass_values[i]))
  print("Uci: " + str(uci_values[i]) + "\n")


# In[16]:


limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [umass_values, uci_values]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Different Algorithms")
label_name = ['c_umass', 'c_uci']
count = 0
for line in y:
  plt.plot(x, line, label = label_name[count])
  count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='best')
plt.show()


# In[21]:


# Show umass graph
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, umass_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[19]:


# Show uci graph
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, uci_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[22]:


limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [coherence_values, umass_values, uci_values]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Different Algorithms")
label_name = ['c_v','c_umass', 'c_uci']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='best')
plt.show()


# In[ ]:





# In[ ]:





# In[17]:


# normalized
model_list[4].show_topics()


# In[18]:


def normalize(x):
    return np.tanh(x)

n_umass = normalize(umass_values)
n_uci = normalize(uci_values)
n_cv = normalize(coherence_values)


# In[19]:


coherence_values
n_cv


# In[20]:


limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [n_cv, n_umass, n_uci]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Different Algorithms")
label_name = ['c_v','c_umass', 'c_uci']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='best')
plt.show()


# In[21]:


# Show n_umass graph
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, n_umass)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("c_umass"), loc='best')
plt.show()


# In[22]:


# Show n_uman_uciss graph
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, n_uci)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("c_uci"), loc='best')
plt.show()


# In[23]:


# Show c_v graph
limit=11; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, n_cv)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("c_v"), loc='best')
plt.show()


# In[47]:


# 5 topics
start = time.time()
test_lda_model = pyLDAvis.gensim.prepare(model_list[3], corpus_tfidf2_bi, dictionary2_bi)
print("Processed time: " + str((time.time() - start)/60))
pyLDAvis.display(test_lda_model)


# # Normalize max min

# In[24]:


def normalize_maxmin(x, min, max):
    return (x - min) / (max - min)
def get_normalized_array(arr):
    return [normalize_maxmin(n, min(arr), max(arr)) for n in arr]


# In[25]:


# get number for cv
nmm_cv = get_normalized_array(n_cv)
nmm_cv


# In[26]:


# get number for umass
nmm_umass = get_normalized_array(n_umass)
nmm_umass


# In[27]:


# get number for uci
nmm_uci = get_normalized_array(n_uci)
nmm_uci


# In[28]:


limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [nmm_cv, nmm_umass, nmm_uci]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Different Algorithms")
label_name = ['c_v','c_umass', 'c_uci']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='best')
plt.show()


# ## Human Judgement

# In[29]:


# maple
maple = [5, 5, 6, 8, 5, 7, 4, 2, 2]
# nick
nick = [5, 6, 7, 9, 6, 7, 5, 4, 4.5]
# alice
alice = [6, 7, 7.5, 9.8, 6, 7, 4, 3, 3]
# jason
jason = [6, 5.8, 6.2, 8.5, 5.4, 5.8, 4.2, 4, 4]
# professor
professor = [5, 5, 7, 8, 7, 8, 5, 5, 5]


# In[30]:


n_maple = get_normalized_array(maple)
n_nick = get_normalized_array(nick)
n_alice = get_normalized_array(alice)
n_jason = get_normalized_array(jason)
n_professor = get_normalized_array(professor)


# In[31]:


# only human judgement
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [n_maple, n_nick, n_alice, n_jason, n_professor]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores by Human Judgement")
label_name = ['person_1','person_2', 'person_3', 'person_4', 'person_5']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()


# In[32]:


# avg of all human judgement
human_judge = []
human_judge.append(n_maple)
human_judge.append(n_nick)
human_judge.append(n_alice)
human_judge.append(n_jason)
human_judge.append(n_professor)


# In[33]:


human_avg = []
for i in range(len(n_maple)):
    _sum = []
    for j in range(len(human_judge)):
        _sum.append(human_judge[j][i])
    human_avg.append(numpy.mean(_sum))
human_avg


# In[34]:


# avg human judgement
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [human_avg]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores by Human Judgement (Average)")
label_name = ['human average']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='best')
plt.show()


# In[35]:


# combine human and machine
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [nmm_cv, nmm_umass, nmm_uci, human_avg]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Different Algorithms and Human Judgement")
label_name = ['c_v','c_umass', 'c_uci', 'human judgement']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()


# # Krippendorff's Value

# In[36]:


# !pip install numpy krippendorff (Krippendorff's alpha)
import krippendorff
kappa = krippendorff.alpha(human_judge)
print(kappa) 


# In[61]:


# avg of all automated metrics
avg_metrics = []
for i in range(len(nmm_cv)):
    avg_metrics.append((nmm_cv[i] + nmm_umass[i] + nmm_uci[i])/3)
avg_metrics


# In[62]:


# avg human judgement
avg_human = []
for i in range(len(human_judge[0])):
    avg_human.append((human_judge[0][i] + human_judge[1][i] + human_judge[2][i] + human_judge[3][i] + human_judge[4][i])/5)
avg_human


# In[76]:


# combine human and machine
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [avg_metrics, human_avg]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Automated Metrics (Average) and Human Judgement (Average)")
label_name = ['average automated metrics', 'human judgement']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()


# In[77]:


# combine human and machine
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [nmm_cv, human_avg]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on CV Metric and Human Judgement (Average)")
label_name = ['average automated metrics', 'human judgement']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()


# In[78]:


# combine human and machine
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [nmm_umass, human_avg]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on Umass Metric and Human Judgement (Average)")
label_name = ['average automated metrics', 'human judgement']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()


# In[79]:


# combine human and machine
limit=11; start=2; step=1;
x = range(start, limit, step) # number of topics
y = [nmm_uci, human_avg]
plt.xlabel("Number of topics")
plt.ylabel("Coherence Scores")
plt.title("Coherence Scores on nmm_uci Metric and Human Judgement (Average)")
label_name = ['average automated metrics', 'human judgement']
count = 0
for line in y:
    plt.plot(x, line, label = label_name[count])
    count += 1
# for i in range(len(y[0])):
#     plt.plot(x,[pt[i] for pt in y],label = 'id %s'%i)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()


# In[63]:


# avg metrics and avg human judge
kappa = krippendorff.alpha([avg_metrics,avg_human])
print(kappa) 


# In[64]:


# cv and avg human judge
kappa = krippendorff.alpha([nmm_cv,avg_human])
print(kappa) 


# In[65]:


# umass and avg human judge
kappa = krippendorff.alpha([nmm_umass,avg_human])
print(kappa) 


# In[66]:


# uci and avg human judge
kappa = krippendorff.alpha([nmm_uci,avg_human])
print(kappa) 


# # Number of document per topic

# In[37]:


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=model_list[3], corpus=corpus_tfidf2_bi, texts=transcript2)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()


# Show
df_dominant_topic.head(10)


# In[38]:


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()


# In[39]:


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics.head()


# In[40]:


topic_counts
# 0 = investi
# 1 = company
# 2 = election
# 3 = percent, marijuana
# 4 = film


# In[41]:


df_dominant_topics[df_dominant_topics['Dominant_Topic'] == 0.0]['Topic_Keywords'].unique().tolist()


# In[42]:


df_dominant_topics[df_dominant_topics['Dominant_Topic'] == 1.0]['Topic_Keywords'].unique().tolist()


# In[43]:


df_dominant_topics[df_dominant_topics['Dominant_Topic'] == 2.0]['Topic_Keywords'].unique().tolist()


# In[44]:


df_dominant_topics[df_dominant_topics['Dominant_Topic'] == 3.0]['Topic_Keywords'].unique().tolist()


# In[45]:


df_dominant_topics[df_dominant_topics['Dominant_Topic'] == 4.0]['Topic_Keywords'].unique().tolist()


# In[46]:


doc_counts = topic_counts.reset_index()
doc_counts.columns = ['Topic_num', 'Doc_counts']
doc_counts = doc_counts.astype(int)
doc_counts


# In[ ]:


# Topic_2 = 15818 docs (['election, president, state, country, vote, campaign, police, government, attack, official'])
# Topic_4 = 11188 docs (['film, woman, video, love, star, movie, like, fan, host, actor'])
# Topic_1 = 7758 docs (['company, percent, health_care, tax, million, climate_change, state, school, plan, woman'])
# Topic_0 = 271 docs (['investigation, email, intelligence, information, probe, campaign, special_counsel, committee, hacking, document'])
# Topic_3= 21 docs (['percent, marijuana, margin_error, based_margin, survey, cannabis, model_based, assumption_wrong, survey_error, error_rest'])

