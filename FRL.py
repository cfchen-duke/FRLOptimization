# FRL.py: learning falling rule lists
# author: Chaofan Chen
#
from __future__ import division

import gmpy
import numpy as np
import copy

from FRLutil import find_caught_instances, find_remain_instances
from curiosity import compute_curiosity

def learn_FRL(X_pos, X_neg, n, w, C, prob_terminate, T, lmda):
    # initialize
    d_ant_best = []
    d_prob_best = []
    
    d_pos_best = []
    d_neg_best = []
    d_obj_best = []
    L_d_over_iters = []    
    L_d_best_over_iters = []
    
    L_d_best = float("inf")
    
    for t in range(T):
        if (t + 1) % 100 == 0:
            print "building FRL %d" % (t + 1)
        
        # available_antecedents does not include the default "null" rule
        available_antecedents = [j for j in range(1,len(X_pos))]
        remaining_pos = copy.deepcopy(X_pos[0])
        remaining_neg = copy.deepcopy(X_neg[0])
        remaining_pos_cnt = gmpy.popcount(remaining_pos)
        remaining_neg_cnt = gmpy.popcount(remaining_neg)
        alpha_last = 1
        d_ant = []
        d_prob = []
        
        d_pos = []
        d_neg = []
        d_obj = []        
        
        L_d = 0
        
        size_candidate_set = []  
        
        while not check_terminating_conditions(alpha_last,
                                               remaining_pos_cnt,
                                               remaining_neg_cnt,
                                               w,C,n):
            terminate = np.random.binomial(1, prob_terminate, 1)[0]
            if terminate:
               break
           
            candidate_antecedents = []
            candidate_prob = []
           
            for j in available_antecedents:
                caught_pos_j,caught_neg_j,ncaught_pos_j,ncaught_neg_j = \
                    find_caught_instances(X_pos[j], X_neg[j], 
                                          remaining_pos, remaining_neg)
                remain_pos_j,remain_neg_j,nremain_pos_j,nremain_neg_j = \
                    find_remain_instances(caught_pos_j, caught_neg_j, 
                                          remaining_pos, remaining_neg)
               
                ncaught_j = ncaught_pos_j + ncaught_neg_j
                nremain_j = nremain_pos_j + nremain_neg_j
               
                if (ncaught_j == 0) or (nremain_j == 0):
                    continue
               
                alpha_antecedent_j = ncaught_pos_j/ncaught_j
               
                if check_antecedent_feasibility(alpha_antecedent_j, alpha_last,
                                                nremain_pos_j, nremain_neg_j,
                                                w):
                    L_j = compute_L(ncaught_pos_j, ncaught_neg_j,
                                    w, C, n)
                    L_dj = L_d + L_j
                    Z_dj = compute_min(nremain_pos_j, nremain_neg_j,
                                       alpha_antecedent_j,
                                       w, C, n)
                    if L_dj + Z_dj < L_d_best:
                        candidate_antecedents.append(j)
                        candidate_prob.append(compute_curiosity(
                                              alpha_antecedent_j, ncaught_pos_j,
                                              remaining_pos_cnt, lmda))
                     
            size_candidate_set.append(len(candidate_antecedents))
            
            if candidate_antecedents:                             
                try:
                    candidate_prob = [c/sum(candidate_prob) \
                                      for c in candidate_prob]
                    r = np.random.choice(np.array(candidate_antecedents),
                                         p=candidate_prob)
                except ZeroDivisionError:
                    r = np.random.choice(np.array(candidate_antecedents))
               
                d_ant.append(r)
                caught_pos_r,caught_neg_r,ncaught_pos_r,ncaught_neg_r = \
                    find_caught_instances(X_pos[r], X_neg[r],
                                          remaining_pos, remaining_neg)
                remain_pos_r,remain_neg_r,nremain_pos_r,nremain_neg_r = \
                    find_remain_instances(caught_pos_r, caught_neg_r,
                                          remaining_pos, remaining_neg)
                alpha_antecedent_r = ncaught_pos_r/(ncaught_pos_r+ncaught_neg_r)
                d_prob.append(alpha_antecedent_r)
               
                d_pos.append(ncaught_pos_r)
                d_neg.append(ncaught_neg_r)
               
                L_r = compute_L(ncaught_pos_r, ncaught_neg_r, w, C, n)
                d_obj.append(L_r)
                L_d = L_d + L_r
               
                available_antecedents.remove(r)
                remaining_pos = remain_pos_r
                remaining_neg = remain_neg_r
                remaining_pos_cnt = nremain_pos_r
                remaining_neg_cnt = nremain_neg_r
                alpha_last = alpha_antecedent_r
            else:
                break
        
        d_ant.append(0)
        remaining_cnt = remaining_pos_cnt + remaining_neg_cnt
        
        if remaining_cnt == 0:
            d_prob.append(0.0)
        else:
            d_prob.append(remaining_pos_cnt/remaining_cnt)
        
        d_pos.append(remaining_pos_cnt)
        d_neg.append(remaining_neg_cnt)
        
        L_else = compute_L(remaining_pos_cnt, remaining_neg_cnt, w, C, n,
                           else_clause = True)
        d_obj.append(L_else)
        L_d = L_d + L_else     
        
        if L_d < L_d_best:
            d_ant_best = copy.deepcopy(d_ant)
            d_prob_best = copy.deepcopy(d_prob)
            d_pos_best = copy.deepcopy(d_pos)
            d_neg_best = copy.deepcopy(d_neg)
            d_obj_best = copy.deepcopy(d_obj)
            L_d_best = L_d
        
        L_d_over_iters.append(L_d)
        L_d_best_over_iters.append(L_d_best)
    
    return d_ant_best, d_prob_best, d_pos_best, d_neg_best, d_obj_best, \
           L_d_best, L_d_over_iters, L_d_best_over_iters

### termination condition checks

def check_terminating_conditions(alpha_last, remaining_pos_cnt,
                                 remaining_neg_cnt, w, C, n):
    if (C >= (min(w*remaining_pos_cnt/n, remaining_neg_cnt/n)) - \
        (((1/alpha_last)-1)*remaining_pos_cnt)/n):                      
       return True
    else:
       return False                              

### feasibility checks

def check_antecedent_feasibility(alpha_antecedent, alpha_last,
                                 nremain_pos_after_antecedent,
                                 nremain_neg_after_antecedent, w):
    nremain = nremain_pos_after_antecedent + nremain_neg_after_antecedent   
    alpha_remain = nremain_pos_after_antecedent/nremain

    if alpha_antecedent <= alpha_last and \
       alpha_antecedent > 1/(1+w) and \
       alpha_remain <= alpha_last:
        return True
    else:
        return False

### computations

def compute_L(ncaught_pos_ant, ncaught_neg_ant, w, C, n, **options):
    # else_clause is a valid option:
    # if else_clause == True, C will not be added to L_ant
    # if else_clause is not supplied, or if else_clause == False,
    # C will be added to L_ant

    else_bool = False

    if "else_clause" in options:
        else_bool = options.get("else_clause")

    if else_bool:
        if (w*ncaught_pos_ant > ncaught_neg_ant):
            # equivalent to alpha_ant > 1/(1+w)
            L_ant = (1/n)*ncaught_neg_ant
        else:
            L_ant = (w/n)*ncaught_pos_ant  
    else:
        if (w*ncaught_pos_ant > ncaught_neg_ant):
            L_ant = (1/n)*ncaught_neg_ant + C
        else:
            L_ant = (w/n)*ncaught_pos_ant + C
    
    return L_ant

def compute_min(nremain_pos, nremain_neg, alpha, w, C, n):
    Z1 = (1/n)*((1/alpha)-1)*nremain_pos + C
    
    return min(Z1, ((w*nremain_pos)/n), (nremain_neg/n))
