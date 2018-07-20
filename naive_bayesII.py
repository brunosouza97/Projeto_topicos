import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
import sklearn as smv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

stopwordsNLTK = nltk.corpus.stopwords.words('english')


dataset = pd.read_csv('text_emotion.csv', error_bad_lines=False)
print(dataset.count())

tweets = dataset['content'].values
classes = dataset['sentiment'].values

def remove_hashtag(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = " ".join(word.strip() for word in re.split('#|_|@', word))
        processed_word_list.append(limpo)

    return processed_word_list
def remove_url(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)
        processed_word_list.append(limpo)

    return processed_word_list

def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower()  # in case they arenet all lower cased
        semStop = [p for p in word.split() if p not in stopwordsNLTK]
        retorno = ' '.join(semStop)
        processed_word_list.append(retorno)
    return processed_word_list

def aplica_stemmer(word_list):
    stemmer = nltk.stem.RSLPStemmer()
    frasesStemming = []
    for word in word_list:
        comStemming = [str(stemmer.stem(p)) for p in word.split()]
        retorno = ' '.join(comStemming)
        frasesStemming.append((retorno))
    return frasesStemming

tweets_limpo = remove_url(tweets)
tweets_limpo1 = remove_hashtag(tweets_limpo)
tweets_sem_stop_word = remove_stopwords(tweets_limpo1)
tweets_com_steming = aplica_stemmer(tweets_sem_stop_word)
#print(tweets_com_steming)
#print(np.unique(classes))
vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets_com_steming)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

testes = ['i hate you']
#t1 = remove_stopwords(testes)
#t2 = aplica_stemmer(t1)
freq_testes = vectorizer.transform(testes)
teste = modelo.predict(freq_testes)
print(teste)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print(metrics.accuracy_score(classes,resultados))
sentimento=['anger','boredom','empty','enthusiasm','fun','happiness','hate','love','neutral','relief','sadness', 'surprise', 'worry']
print(metrics.classification_report(classes,resultados,labels=np.unique(resultados)))
print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), )