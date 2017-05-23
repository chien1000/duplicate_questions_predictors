#! python3
# -*- coding:utf-8 -*-
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
import pickle
VECTORIZER = None
WORD2VEC = None

def initialize(data, token_data):
    global VECTORIZER
    global WORD2VEC
    if data:
        VECTORIZER = TfidfVectorizer(stop_words='english')
        VECTORIZER.fit(data)
        pickle.dump(VECTORIZER.vocabulary_, open("vocabulary.pkl","wb"))

        WORD2VEC = Word2Vec(token_data, size=300, window=5, min_count=1, workers=4)
        WORD2VEC.save("word2vec_model.bin")
    else :
        VECTORIZER = TfidfVectorizer(decode_error="replace", vocabulary=pickle.load(open("vocabulary.pkl", "rb")))
        WORD2VEC = KeyedVectors.load("word2vec_model.bin")

def jaccard(s1,s2):
    u = s1.union(s2)
    i = s1.intersection(s2)
    j = len(i) / len(u)

    return j

class Qpair(object):
    def __init__(self,q1,q2,is_duplicate):
       self.q1 = q1
       self.q2 = q2
       self.is_duplicate = is_duplicate
       # tokenize and pos tag
       self.q1_tokenized = nltk.pos_tag(nltk.word_tokenize(self.q1.lower()))
       self.q2_tokenized = nltk.pos_tag(nltk.word_tokenize(self.q2.lower()))
       
    def count_same_words(self,normalize=True):
        sq1 = set([tup[0] for tup in self.q1_tokenized])
        sq2 = set([tup[0] for tup in self.q2_tokenized])
        if normalize: #TODO: how to normalize
            ocount = jaccard(sq1,sq2)
        else:
            overlap = sq1.intersection(sq2)
            ocount = len(overlap)
        return ocount

    def get_pos_info(self):
        values = []
        for target_pos in ["N","V","J"]:
            s1 = set( [tup[0] for tup in self.q1_tokenized if tup[1].startswith(target_pos)] )
            s2 = set( [tup[0] for tup in self.q2_tokenized if tup[1].startswith(target_pos)] )
            if s1 and s2:
                jscore = jaccard(s1,s2)
            else:
                jscore = 1
            values.append(jscore)
        return values
    
    def get_tfidf_diff(self):
        global VECTORIZER
        m = VECTORIZER.transform([self.q1,self.q2])
        diff = np.absolute(m[0,] - m[1,])
        return diff

    def get_wordvector_diff(self):
        pass

    def get_features(self):
        fv = []
        # words_count = self.count_same_words()
        # fv.append(words_count)
        
        # pos_info = self.get_pos_info()
        # fv.extend(pos_info)


        # fv = np.array(fv)
        fv = self.get_tfidf_diff()
        return fv
    
    @staticmethod
    def get_feature_names():
        # return ["word_jaccard", "N_jaccard", "V_jaccard", "J_jaccard"]
        global VECTORIZER        
        name = VECTORIZER.get_feature_names()
        return name

def main():
    q1 = 'chien is sleepy and hungry'
    q2 = 'hi my name is chien'
    qp = Qpair(q1,q2,1)
    initialize([q1,q2])
    print(qp.q1_tokenized)
    print(qp.q2_tokenized)
    print("same words count {}, non normalized {}".format(qp.count_same_words(),qp.count_same_words(False)))
    print("pos info N {v[0]} V {v[1]} J {v[2]}".format(v=qp.get_pos_info()))
    print(qp.get_features())

if __name__ == '__main__':
    main()