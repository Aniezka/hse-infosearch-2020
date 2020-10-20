### реализуйте эту функцию ранжирования
import collections
import string
import math
import pandas as pd
import numpy as np
import json

from tqdm import tqdm
from pymorphy2 import MorphAnalyzer
from razdel import tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, KeyedVectors


print("Load matrices, please wait....")
morph = MorphAnalyzer()
stop = set(stopwords.words('russian'))

def my_preprocess(text: str):
    text = str(text)
    text = text.replace("\n", " ").replace('/', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = list(tokenize(text))
    lemm = [morph.parse(i.text)[0].normal_form for i in tokenized_text]
    words = [i for i in lemm if i not in stop]
    return " ".join(words)

answ = pd.read_excel("answers_base.xlsx")
answ.rename(columns={'Текст вопросов': 'text', 'Номер связки': 'join_num'}, inplace=True)

train_quest = pd.read_csv("train_processed.csv", sep=';')
train_quest_processed = [str(i) for i in train_quest['normal_text'].values]

def get_answ(num):
    all_answ = answ[answ.join_num == float(num)]['Текст ответа']
    return [(num, str(all_answ[0]))]
#--------------------------------------------------------------------------------------------TF-IDF block

vectorizer_tf_idf = TfidfVectorizer()

vectorizer_tf_idf.fit(train_quest_processed)
# train matrix
X_train_tf_idf = vectorizer_tf_idf.transform(train_quest_processed).toarray()

def get_query_tf_idf(query):
    query_vec = vectorizer_tf_idf.transform([query]).toarray()
    search_result = []
    rating = X_train_tf_idf.dot(query_vec.T)
    rating = rating.flatten()
    rating = np.argpartition(rating, -20)[-20:]
    for pred in rating:
        if not math.isnan(train_quest.iloc[pred].join_num):
            search_result.append((str(int(train_quest.iloc[pred].join_num)), train_quest.iloc[pred].text))
    return search_result

#--------------------------------------------------------------------------------------------BM25 block

def vec_bm25(doc, dict_words):
    vec = np.zeros((1, len(dict_words)))
    for word in doc.split(" "):
        if word in dict_words.keys():
            vec[0, dict_words[word][-1]] = 1
    return vec

with open('bm25_inverse_dict.json', 'r') as fp:
    inverse_dict = json.load(fp)
X_train_bm25 = np.load('X_train_bm25.npy')

def get_query_bm25(query):
    query_vec = vec_bm25(query, inverse_dict)
    search_result = []
#    print(X_train_bm25.dot(query_vec.T).flatten().shape)
    for pred in np.argpartition(X_train_bm25.dot(query_vec.T).flatten(), -20)[-20:]:
#    for pred in np.array(X_train_bm25.dot(query_vec.T).argmax(axis=0))[:20]:
#        pred = np.array(X_train_bm25.dot(query_vec.T).argmax(axis=0))[0]
        if not math.isnan(train_quest.iloc[pred].join_num):
            search_result.append((str(int(train_quest.iloc[pred].join_num)), train_quest.iloc[pred].text))
    return search_result

#--------------------------------------------------------------------------------------------w2v block

model_file = 'araneum_none_fasttextcbow_300_5_2018.model'
model = KeyedVectors.load(model_file)

def normalize_vec(vec):
    return vec / np.linalg.norm(vec)

def create_doc_vector(text):
    # создаем вектор-маску
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    # если слово есть в модели, берем его вектор
    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lemmas_vectors[idx] = normalize_vec(model[lemma])
    # проверка на случай, если на вход пришел пустой массив
    if lemmas_vectors.shape[0] is not 0:
        return normalize_vec(np.mean(lemmas_vectors, axis=0))
    else:
        np.zeros((model.vector_size,))

X_train_mean_w2v = np.load("X_train_mean_w2v.npy")

def get_query_w2v(query):
    query_vec = create_doc_vector(query)
    search_result = []
    for pred in np.argpartition(X_train_mean_w2v.dot(query_vec.T), -20, axis=0)[-20:]:
#    pred = X_train_mean_w2v.dot(query_vec.T).argpartition(a, -4)[-4:]
        if not math.isnan(train_quest.iloc[pred].join_num):
            search_result.append((str(int(train_quest.iloc[pred].join_num)), train_quest.iloc[pred].text))
    return search_result

#--------------------------------------------------------------------------------------------d2v block

def create_doc_matrix(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    for idx, lemma in enumerate(lemmas):
        if lemma in model.wv:
            lemmas_vectors[idx] = normalize_vec(model[lemma])

    return lemmas_vectors

def search(docs, query, reduce_func=np.max, axis=0):
    sims = []
    for doc in docs:
        sim = doc.dot(query.T)
        sim = reduce_func(sim, axis=axis)
        sims.append(sim.sum())
    return np.argpartition(sims, -20)[-20:]
#    return np.argmax(sims)

X_train_d2v = np.load("X_train_d2v.npy", allow_pickle=True)

def get_query_d2v(query):
    query_vec = create_doc_matrix(query)
    search_result = []
    for pred in search(X_train_d2v, query_vec):
#    pred = search(X_train_d2v, query_vec)
        if not math.isnan(train_quest.iloc[pred].join_num):
            search_result.append((str(int(train_quest.iloc[pred].join_num)), train_quest.iloc[pred].text))
    return search_result

print("Matrices loaded!")
