# main.py: main script demonstrating Algorithm FRL and Algorithm softFRL
# author: Chaofan Chen
#
from __future__ import division
from data import load_data, load_labels
from antecedents import mine_antecedents
from FRL import learn_FRL
from softFRL import learn_softFRL
from display import display_rule_list, display_softFRL

def main():
    dataset = "bank-full"
    path = "datasets/" + dataset + "/all/"
    fname_data = path + dataset + "_all.X"
    fname_label = path + dataset + "_all.Y"
    
    data = load_data(fname_data)
    Y = load_labels(fname_label)
    
    # mine rules
    print "mining rules using FP-growth"
    minsupport = 10
    max_predicates_per_ant = 2
    X_pos,X_neg,nantecedents,antecedent_len,antecedent_set = \
        mine_antecedents(data,Y,minsupport,max_predicates_per_ant)
    
    n = len(data)
    
    # learn a falling rule list from the training data
    # set the parameters of Algorithm FRL
    w = 7
    C = 0.000001
    prob_terminate = 0.01
    T_FRL = 3000
    
    # set the parameters of Algorithm softFRL
    C1 = 0.5
    T_softFRL = 6000
    
    # set the parameter of the curiosity function
    lmda = 0.8
    
    # train a falling rule list
    print "running algorithm FRL on bank-full"
    FRL_rule, FRL_prob, FRL_pos_cnt, FRL_neg_cnt, FRL_obj_per_rule, FRL_Ld, \
        FRL_Ld_over_iters, FRL_Ld_best_over_iters = \
        learn_FRL(X_pos, X_neg, n, w, C, prob_terminate, T_FRL, lmda)
    
    print "FRL learned:"
    display_rule_list(FRL_rule, FRL_prob, antecedent_set, FRL_pos_cnt, FRL_neg_cnt,
                      FRL_obj_per_rule, FRL_Ld)
    
    print "running algorithm softFRL on bank-full"
    softFRL_rule, softFRL_prob, softFRL_pos_cnt, softFRL_neg_cnt, \
        softFRL_pos_prop, softFRL_obj_per_rule, softFRL_Ld, \
        softFRL_Ld_over_iters, softFRL_Ld_best_over_iters = \
        learn_softFRL(X_pos, X_neg, n, w, C, C1, prob_terminate,
                      T_softFRL, lmda)
    
    print "softFRL learned:"    
    display_softFRL(softFRL_rule, softFRL_prob, antecedent_set,
                    softFRL_pos_cnt, softFRL_neg_cnt, softFRL_pos_prop,
                    softFRL_obj_per_rule, softFRL_Ld)             

if __name__ == '__main__':
    main()
