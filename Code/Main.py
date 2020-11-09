import os,random,math
from nltk import word_tokenize as tokenize
import operator
import pandas as pd, csv
from nltk import word_tokenize as tokenize
from scc import *
from model import *
from ab_model import *

#Directory of the training data
TRAINING_DIR="sentence-completion/Holmes_Training_Data"
#test questions file
QUESTIONS_DIR = "sentence-completion/testing_data.csv"
#test answers file
ANSWERS_DIR = "sentence-completion/test_answer.csv"
#the question file used to test comlpexity
COMPLEX_DIR = "sentence-completion/concat_questions/all_questions.txt"

#used to test for unknown threshold
for i in range(5):
    for x in range(9):
        lm = language_model(TRAINING_DIR, max_files = ((x+1)*10))
        lm.train(unk_thresh = i)
        print("-----------------unigram---------------- files:{} --- ukn_thresh: {} ".format(((x+1)*10), i))
        print("method: unigram, score: {}".format(SCC.predict_and_score(model = lm,method = "unigram")))
        print("-----------------no AD---------------- files:{} --- ukn_thresh: {} ".format(((x+1)*10), i))
        print("---bi----")
        print("method: bigram, score: {}".format(SCC.predict_and_score(model = lm,method = "bigram")))
        print("---tri----")
        print("method: trigram, score: {}".format(SCC.predict_and_score(model = lm,method = "trigram")))

        lm = ab_model(TRAINING_DIR, max_files = ((x+1)*10))
        lm.train(unk_thresh = i)

        print("-----------------AD-----------------files:{} --- ukn_thresh: {} ".format(((x+1)*10), i))
        print("---bi----")
        print("method: bigram, score: {}".format(SCC.predict_and_score(model = lm,method = "bigram")))
        print("---tri----")
        print("method: trigram, score: {}".format(SCC.predict_and_score(model = lm,method = "trigram")))

#used to test for full training data use
i =2
lm = language_model(TRAINING_DIR, limit_files = False)
lm.train(unk_thresh = i)
print("-----------------unigram---------------- files:{} --- ukn_thresh: {} ".format(522, i))
print("method: unigram, score: {}".format(SCC.predict_and_score(model = lm,method = "unigram")))
print("-----------------no AD---------------- files:{} --- ukn_thresh: {} ".format(522, i))
print("---bi----")
print("method: bigram, score: {}".format(SCC.predict_and_score(model = lm,method = "bigram")))
print("---tri----")
print("method: trigram, score: {}".format(SCC.predict_and_score(model = lm,method = "trigram")))

lm = ab_model(TRAINING_DIR, limit_files = False)
lm.train(unk_thresh = i)

print("-----------------AD-----------------files:{} --- ukn_thresh: {} ".format(522, i))
print("---bi----")
print("method: bigram, score: {}".format(SCC.predict_and_score(model = lm,method = "bigram")))
print("---tri----")
print("method: trigram, score: {}".format(SCC.predict_and_score(model = lm,method = "trigram")))

#used to test for full training data used with trigram method 2
i =2
lm = language_model(TRAINING_DIR, limit_files = False)
lm.train(unk_thresh = i)

print("-----------------no AD---------------- files:{} --- ukn_thresh: {} ".format(522, i))
print("---tri2----")
print("method: trigram, score: {}".format(SCC.predict_and_score(model = lm,method = "trigram2")))

lm = ab_model(TRAINING_DIR, limit_files = False)
lm.train(unk_thresh = i)

print("-----------------AD-----------------files:{} --- ukn_thresh: {} ".format(522, i))
print("---tri2----")
print("method: trigram, score: {}".format(SCC.predict_and_score(model = lm,method = "trigram2")))
"""
#used to produce data stats
lm = ab_model(TRAINING_DIR, limit_files = False)
lm.train(unk_thresh = 2)
#lm.data_stats()

print("Complexity, AB_model, unigram, all files:{}".format(lm.compute_perplexity([COMPLEX_DIR]), method = "unigram"))
print("Complexity, AB_model, trigram, all files:{}".format(lm.compute_perplexity([COMPLEX_DIR]), method = "trigram"))
