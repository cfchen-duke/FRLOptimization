# softFRL.py: learning softly falling rule lists
# author: Chaofan Chen
#
from __future__ import division

import gmpy
import numpy as np
import copy

from FRLutil import find_caught_instances, find_remain_instances
from curiosity import compute_curiosity_softFRL

def learn_softFRL(X_pos, X_neg, n, w, C, C1, prob_terminate, T, lmda):
    # initialize the current best rule list
    # Note: we can initialize the current best rule list to, say, {R1, else}
    d_ant_best = []
    d_pos_prop_best = []
    d_prob_best = []

    d_pos_cnt_best = []
    d_neg_cnt_best = []
    d_obj_best = []
    
    L_d_over_iters = []
    L_d_best_over_iters = []
    
    L_d_best = float("inf")
    
    for t in range(T):
        if (t + 1) % 500 == 0:
            print "building rule list %d" % (t + 1)
        
        # available_antecedents does not include the default "null" rule
        available_antecedents = [j for j in range(1,len(X_pos))]
        remaining_pos = copy.deepcopy(X_pos[0])
        remaining_neg = copy.deepcopy(X_neg[0])
        remaining_pos_cnt = gmpy.popcount(remaining_pos)
        remaining_neg_cnt = gmpy.popcount(remaining_neg)
        alpha_min = 1
        d_ant = []
        d_pos_prop = []

        d_pos_cnt = []
        d_neg_cnt = []        
        d_obj = []
        
        L_d = 0
        
        size_candidate_set = []        
        
        while True:
            terminate = np.random.binomial(1, prob_terminate, 1)[0]
            if (terminate):
                break
            
            candidate_antecedents = []
            candidate_prob = []
            should_terminate = []
            
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
               
                alpha_ant_j = ncaught_pos_j/ncaught_j
                alpha_min_dj = min(alpha_min, alpha_ant_j)

                L_j = compute_L_ant_softFRL(ncaught_pos_j, ncaught_neg_j, 
                                            alpha_min,
                                            w, C, C1, n)
                L_dj = L_d + L_j
                b1_dj = compute_lb_not_term(nremain_pos_j, nremain_neg_j,
                                            alpha_min_dj,
                                            w, C, C1, n)
                # T_dj: contribution to the objective if we terminate
                T_dj = compute_T(nremain_pos_j, nremain_neg_j, alpha_min_dj,
                                 w, C1, n)
                Z_dj = min(b1_dj, T_dj)
                
                if (L_dj + Z_dj < L_d_best):
                    candidate_antecedents.append(j)
                    candidate_prob.append(compute_curiosity_softFRL(
                                          alpha_ant_j, alpha_min,
                                          ncaught_pos_j, remaining_pos_cnt,
                                          lmda))
                    if (T_dj <= b1_dj):
                        should_terminate.append(j)
                
            size_candidate_set.append(len(candidate_antecedents))
                        
            if (candidate_antecedents):
                try:
                    candidate_prob = [c/sum(candidate_prob) \
                                      for c in candidate_prob]
                    r = np.random.choice(np.array(candidate_antecedents),
                                         p=candidate_prob)
                except (ZeroDivisionError, ValueError):
                    r = np.random.choice(np.array(candidate_antecedents))
                
                d_ant.append(r)
                caught_pos_r,caught_neg_r,ncaught_pos_r,ncaught_neg_r = \
                    find_caught_instances(X_pos[r], X_neg[r],
                                          remaining_pos, remaining_neg)
                remain_pos_r,remain_neg_r,nremain_pos_r,nremain_neg_r = \
                    find_remain_instances(caught_pos_r, caught_neg_r,
                                          remaining_pos, remaining_neg)
                alpha_ant_r = ncaught_pos_r/(ncaught_pos_r+ncaught_neg_r)
                d_pos_prop.append(alpha_ant_r)

                d_pos_cnt.append(ncaught_pos_r)
                d_neg_cnt.append(ncaught_neg_r)
                
                L_r = compute_L_ant_softFRL(ncaught_pos_r, ncaught_neg_r,
                                            alpha_min,
                                            w, C, C1, n)
                d_obj.append(L_r)
                L_d = L_d + L_r
                
                available_antecedents.remove(r)
                remaining_pos = remain_pos_r
                remaining_neg = remain_neg_r
                remaining_pos_cnt = nremain_pos_r
                remaining_neg_cnt = nremain_neg_r
                alpha_min = min(alpha_min, alpha_ant_r)
                
                if (r in should_terminate):
                    break
            else:
                break
        
        # terminate the rule list
        d_ant.append(0)
        remaining_cnt = remaining_pos_cnt + remaining_neg_cnt
        
        if (remaining_cnt == 0):
            d_pos_prop.append(0.0)
        else:
            d_pos_prop.append(remaining_pos_cnt/remaining_cnt)
        
        d_pos_cnt.append(remaining_pos_cnt)
        d_neg_cnt.append(remaining_neg_cnt)
        
        L_else = compute_L_ant_softFRL(remaining_pos_cnt, remaining_neg_cnt,
                                       alpha_min,
                                       w, C, C1, n, else_clause = True)
        d_obj.append(L_else)
        L_d = L_d + L_else
        
        if (L_d < L_d_best):
            d_ant_best = copy.deepcopy(d_ant)
            d_pos_prop_best = copy.deepcopy(d_pos_prop)
            d_pos_cnt_best = copy.deepcopy(d_pos_cnt)
            d_neg_cnt_best = copy.deepcopy(d_neg_cnt)
            d_obj_best = copy.deepcopy(d_obj)
            L_d_best = L_d
        
        L_d_over_iters.append(L_d)
        L_d_best_over_iters.append(L_d_best)

    d_prob_best = make_falling(d_pos_prop_best)
    
    return d_ant_best, d_prob_best, d_pos_cnt_best, d_neg_cnt_best, \
           d_pos_prop_best, d_obj_best, L_d_best, L_d_over_iters, \
           L_d_best_over_iters

### function for computing the objective
def compute_L_ant_softFRL(ncaught_pos_ant, ncaught_neg_ant, alpha_min,
                          w, C, C1, n, **options):
    # else_clause is a valid option:
    # if else_clause == True, C will not be added to L_ant
    # if else_clause is not supplied, or if else_clause == False,
    # C will be added to L_ant

    ncaught_ant = ncaught_pos_ant + ncaught_neg_ant

    if (ncaught_ant):
        alpha_ant = ncaught_pos_ant/ncaught_ant
        
        if (w*ncaught_pos_ant > ncaught_neg_ant):
            # equivalent to alpha_ant > 1/(1+w)
            if (alpha_ant <= alpha_min):
                L_ant = ncaught_neg_ant/n
            else:
                L_ant = (ncaught_neg_ant/n) + (C1*(alpha_ant - alpha_min))
        else:
            if (alpha_ant <= alpha_min):
                L_ant = (w*ncaught_pos_ant)/n
            else:
                L_ant = ((w*ncaught_pos_ant)/n) + (C1*(alpha_ant - alpha_min))
        
    else:
        L_ant = 0.0

    else_bool = False

    if "else_clause" in options:
        else_bool = options.get("else_clause")
    
    if not else_bool:
        L_ant = L_ant + C
    
    return L_ant

def compute_lb_not_term(nremain_pos, nremain_neg, alpha_min,
                        w, C, C1, n):
    if (nremain_pos == 0) or (nremain_neg == 0):
        return float("inf")
    
    alpha_remain = nremain_pos/(nremain_pos + nremain_neg)
    
    inf_g = compute_inf_g(nremain_pos, nremain_neg, alpha_min,
                          w, C, C1, n)
    try:    
        b1 = (((1/alpha_min)-1)*nremain_pos)/n + C
        if (alpha_remain >= alpha_min):
            b1 = b1 + (w*nremain_pos)/n + C1*(alpha_remain - alpha_min)
    except ZeroDivisionError:
        assert(n != 0)
        # alpha_min == 0
        b1 = float("inf")
    
    return min(b1, inf_g)

def compute_inf_g(nremain_pos, nremain_neg, alpha_min,
                  w, C, C1, n):
    alpha_remain = nremain_pos/(nremain_pos + nremain_neg)
    zeta = max(alpha_min, alpha_remain, 1/(1+w))
    g_stationary = np.sqrt(nremain_pos/(C1*n))    
    
    if (g_stationary > zeta) and \
       (g_stationary <= 1):
        b = compute_g(g_stationary, nremain_pos, alpha_min,
                      C, C1, n)
    else:
        b = min(compute_g(zeta, nremain_pos, alpha_min, C, C1, n),
                compute_g(1, nremain_pos, alpha_min, C, C1, n))
    
    return b

def compute_T(nremain_pos, nremain_neg, alpha_min, w, C1, n):
    return compute_L_ant_softFRL(nremain_pos, nremain_neg, alpha_min,
                                  w, 0, C1, n, else_clause = True)

def compute_g(beta, nremain_pos, alpha_min, C, C1, n):
    loss = ((1/beta)-1)*nremain_pos/n
    
    return loss + C + C1*(beta - alpha_min)

### function for processing a rule list into a falling rule list
def make_falling(d_pos_prop):
    d_prob = copy.deepcopy(d_pos_prop)
    
    prob_min = 1.0
    for (i, prob) in enumerate(d_prob):
        if (prob <= prob_min):
            prob_min = prob
        else:
            d_prob[i] = prob_min
    
    return d_prob
