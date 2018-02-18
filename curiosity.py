# curiosity.py: curiosity functions
# author: Chaofan Chen
#
from __future__ import division

def compute_curiosity(alpha_rule, ncaught_pos_rule,
                      nremain_pos_before_rule, lmda):
    curiosity = (lmda*alpha_rule) + (
                 (1-lmda)*(ncaught_pos_rule/nremain_pos_before_rule))
    return curiosity

def compute_curiosity_softFRL(alpha_rule, alpha_min, ncaught_pos_rule,
                      nremain_pos_before_rule, lmda):
    curiosity = (lmda*max(0, min(alpha_rule, (1.01/0.01)*alpha_min - (1/0.01)*alpha_rule))) + (
                 (1-lmda)*(ncaught_pos_rule/nremain_pos_before_rule))
    return curiosity
