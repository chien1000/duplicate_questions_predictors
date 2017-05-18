#! python3
# -*- coding:utf-8 -*-
import numpy as np
import nltk

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

    def get_features(self):
        words_count = self.count_same_words()
        fv = [words_count]
        
        pos_info = self.get_pos_info()
        fv.extend(pos_info)

        fv = np.array(fv)
        return fv
    
    @staticmethod
    def get_feature_names():
        return ["word_jaccard", "N_jaccard", "V_jaccard", "J_jaccard"]

def main():
    q1 = 'chien is sleepy and hungry'
    q2 = 'hi my name is chien'
    qp = Qpair(q1,q2,1)
    print(qp.q1_tokenized)
    print(qp.q2_tokenized)
    print("same words count {}, non normalized {}".format(qp.count_same_words(),qp.count_same_words(False)))
    print("pos info N {v[0]} V {v[1]} J {v[2]}".format(v=qp.get_pos_info()))
    print(qp.get_features())

if __name__ == '__main__':
    main()