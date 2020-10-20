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


print("Create matrices, please wait....")
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
#     return words
    return " ".join(words)
    
answers_df = pd.read_excel("answers_base.xlsx")
questions_df = pd.read_excel("queries_base.xlsx")

answers_df.rename(columns={'Текст вопросов': 'text', 'Номер связки': 'join_num'}, inplace=True)
questions_df.rename(columns={'Текст вопроса': 'text', 'Номер связки\n': 'join_num'}, inplace=True)

train_quest = pd.concat([answers_df, questions_df])

train_quest_processed = []

for quest in tqdm(train_quest['text']):
    train_quest_processed.append(my_preprocess(quest))
    
train_quest['normal_text'] = train_quest_processed
train_quest.to_csv("train_processed.csv", sep=';')

#--------------------------------------------------------------------------------------------TF-IDF block

vectorizer_tf_idf = TfidfVectorizer()

vectorizer_tf_idf.fit(train_quest_processed)
# train matrix
X_train_tf_idf = vectorizer_tf_idf.transform(train_quest_processed)

#--------------------------------------------------------------------------------------------BM25 block

k = 2.0
b = 0.75

def get_inverse_dict(mat):
    d = dict()
    mat = mat.toarray()
    for ind, word in enumerate(count_vectorizer.get_feature_names()):
        d[word] = [int(sum(mat[:, ind]))]
        for ind_j, doc_ind in enumerate(mat[:, ind].tolist()):
            if doc_ind != 0:
                d[word].append(doc_ind)
        d[word].append(ind)
    return d

def bm25_vectorizer(tf_val, len_d, corpus_len, nq):
    IDF = np.log((corpus_len-nq+0.5) / (nq+0.5))
    TF = (tf_val * (k+1)) / (tf_val + k * (1-b+b*(len_d / avrdl)))
    return TF * IDF


corpus_len = len(train_quest_processed)
avrdl = sum([len(i.split(" ")) for i in train_quest_processed]) / corpus_len

count_vectorizer = CountVectorizer()
X_train = count_vectorizer.fit_transform(train_quest_processed)
inverse_dict = get_inverse_dict(X_train)

X_train_bm25 = np.zeros((corpus_len, len(inverse_dict)))

for ind, doc in enumerate(train_quest_processed):
    tokens = doc.split(" ")
    tf_values = collections.Counter(tokens)
    len_d = len(tokens)
    for word in tokens:
        if word not in inverse_dict.keys():
            continue
        X_train_bm25[ind, inverse_dict[word][-1]] = bm25_vectorizer(tf_values[word],
                                                         len_d,
                                                         corpus_len,
                                                         len(inverse_dict[word]) - 1)

with open('bm25_inverse_dict.json', 'w') as fp:
    json.dump(inverse_dict, fp)
np.save('X_train_bm25.npy', X_train_bm25)

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

# train matrix
X_train_mean_w2v = np.array([create_doc_vector(text) for text in train_quest_processed])
np.save("X_train_mean_w2v.npy", X_train_mean_w2v)

#--------------------------------------------------------------------------------------------d2v block

def create_doc_matrix(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lemmas_vectors[idx] = normalize_vec(model[lemma])

    return lemmas_vectors

X_train_d2v = np.array([create_doc_matrix(text) for text in train_quest_processed])
np.save("X_train_d2v.npy", X_train_d2v, allow_pickle=True)

print("Matrices created!")
