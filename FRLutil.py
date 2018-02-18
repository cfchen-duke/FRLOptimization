# FRLutil.py: utilities for finding instances and getting probability estimates
# author: Chaofan Chen
#
from __future__ import division
import gmpy
import numpy as np

def find_caught_instances(X_pos_rule, X_neg_rule, remain_pos, remain_neg):
    caught_pos_rule = X_pos_rule & remain_pos   
    caught_neg_rule = X_neg_rule & remain_neg
    
    ncaught_pos_rule = gmpy.popcount(caught_pos_rule)
    ncaught_neg_rule = gmpy.popcount(caught_neg_rule)
    
    return caught_pos_rule, caught_neg_rule, ncaught_pos_rule, ncaught_neg_rule

def find_remain_instances(caught_pos_rule, caught_neg_rule,
                          remain_pos, remain_neg):
    remain_pos_rule = remain_pos - caught_pos_rule
    remain_neg_rule = remain_neg - caught_neg_rule
    
    nremain_pos_rule = gmpy.popcount(remain_pos_rule)
    nremain_neg_rule = gmpy.popcount(remain_neg_rule)
    
    return remain_pos_rule, remain_neg_rule, nremain_pos_rule, nremain_neg_rule
    
def get_probability_estimates(Xtest, ntest, d_rule, d_prob):
    # Xtest[j] is the set of test data points that satisfy rule j
    # ntest is the number of test data points
    # d_rule, d_prob is the falling rule list
    remaining = set(range(ntest))
    prob_test = -1*np.ones(ntest)
    for i, j in enumerate(d_rule):
        caught = remaining.intersection(Xtest[j])
        prob_test[list(caught)] = d_prob[i] 
        remaining = remaining.difference(caught)
    if prob_test.min() < 0:
        # this means that some data points are not classified
        # by the falling rule list
        raise Exception
    return prob_test

def compute_accuracy(y_true, y_score, threshold):
    assert len(y_true) == len(y_score)    
    n = len(y_true)  
    accuracy = 0
    for i, score in enumerate(y_score):
        if score >= threshold and y_true[i] == 1:
            accuracy = accuracy + 1
        elif score < threshold and y_true[i] == 0:
            accuracy = accuracy + 1
    
    accuracy = accuracy/n    
    return accuracy

def compute_weighted_loss_from_prob(y_true, y_score, threshold, w1):
    assert len(y_true) == len(y_score)    
    n = len(y_true)
    total_pos = sum(y_true)
    total_neg = n - total_pos
    loss = 0
    false_pos = 0
    false_neg = 0
    for i, score in enumerate(y_score):
        if score >= threshold and y_true[i] == 0:
            loss = loss + 1
            false_pos = false_pos + 1
        elif score < threshold and y_true[i] == 1:
            loss = loss + w1
            false_neg = false_neg + 1
    loss = loss/n
    return loss, false_pos/total_neg, (total_pos-false_neg)/total_pos

def compute_weighted_loss_from_label(y_true, y_pred, w1):
    assert len(y_true) == len(y_pred)    
    n = len(y_true)
    total_pos = sum(y_true)
    total_neg = n - total_pos
    loss = 0
    false_pos = 0
    false_neg = 0
    for i, pred in enumerate(y_pred):
        if y_pred[i] == 1 and y_true[i] == 0:
            loss = loss + 1
            false_pos = false_pos + 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            loss = loss + w1
            false_neg = false_neg + 1
    loss = loss/n
    return loss, false_pos/total_neg, (total_pos-false_neg)/total_pos
