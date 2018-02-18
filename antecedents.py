# antecedents.py: mining antecedents from data using fpgrowth
# author: Chaofan Chen
#
from fim import fpgrowth
from collections import Counter
import numpy as np

def mine_antecedents(data,Y,minsupport,max_predicates_per_antecedent):
    # data is the training data
    # Y is the training labels: 1 for positive and 0 for negative
    # minsupport is an integer percentage (e.g. 10 for 10%)
    # max_predicates_per_antecedent is the maximum number of predicates in a rule
    # mine the rule set
    n = len(data)
    data_pos = [x for i,x in enumerate(data) if Y[i] == 1]
    data_neg = [x for i,x in enumerate(data) if Y[i] == 0]
    assert len(data_pos)+len(data_neg) == n
    
    antecedent_set = [r[0] for r in fpgrowth(data_pos, supp=minsupport,
               zmax=max_predicates_per_antecedent)]
    antecedent_set.extend([r[0] for r in fpgrowth(data_neg,supp=minsupport,
                    zmax=max_predicates_per_antecedent)])
    antecedent_set = list(set(antecedent_set))
    print len(antecedent_set),'rules mined'
    # form the rule-versus-data set
    # X_pos[j] is the set of positive data points that satisfy rule j
    # X_neg[j] is the set of negative data points that satisfy rule j
    X_pos = [0 for j in range(len(antecedent_set)+1)]
    X_neg = [0 for j in range(len(antecedent_set)+1)]
    # X_pos[0] (X_neg[0]) is the set of all positive (negative) data points
    X_pos[0] = sum([1<<i for i,x in enumerate(data) if Y[i] == 1])
    X_neg[0] = sum([1<<i for i,x in enumerate(data) if Y[i] == 0])
    for (j,antecedent) in enumerate(antecedent_set):
        X_pos[j+1] = sum([1<<i for (i,xi) in enumerate(data) \
                          if Y[i] == 1 and set(antecedent).issubset(xi)])
        X_neg[j+1] = sum([1<<i for (i,xi) in enumerate(data) \
                          if Y[i] == 0 and set(antecedent).issubset(xi)])
    # form antecedent_len and nantecedents
    antecedent_len = [0]
    for antecedent in antecedent_set:
        antecedent_len.append(len(antecedent))
    nantecedents = Counter(antecedent_len)
    antecedent_len = np.array(antecedent_len)
    antecedent_set_all = ['null']
    antecedent_set_all.extend(antecedent_set)
    return X_pos,X_neg,nantecedents,antecedent_len,antecedent_set_all
