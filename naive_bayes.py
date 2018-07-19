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

stopwordsNLTK = nltk.corpus.stopwords.words('portuguese')
stopwordsNLTK.append('vou')
stopwordsNLTK.append('tão')
stopwordsNLTK.append('não')

dataset = pd.read_csv('Base_teste.csv', error_bad_lines=False)
print(dataset.count())

tweets = dataset['Text'].values
classes = dataset['Classificacao'].values
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


tweets_sem_stop_word = remove_stopwords(tweets)
tweets_com_steming = aplica_stemmer(tweets_sem_stop_word)
print(tweets_com_steming)
vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets_sem_stop_word)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

testes = ['eu odeio você','eu amo você','você me da nojo','não vá ali', 'é perigoso']

freq_testes = vectorizer.transform(testes)
teste = modelo.predict(freq_testes)
#print(teste)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print(metrics.accuracy_score(classes,resultados))
sentimento=['Alegria','Medo','Desprezo','Tristeza','Raiva','Desgosto']
print(metrics.classification_report(classes,resultados,labels=np.unique(resultados)))
print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), )