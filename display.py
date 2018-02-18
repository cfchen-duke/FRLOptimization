# display.py: display functions
# author: Chaofan Chen
#
def display_rule_list(d_rule, d_prob, ruleset, d_pos, d_neg, d_obj, L_d):
    for i in range(len(d_rule)):
        if i == 0:
            print "if %s, then prob. = %f (+: %d, -: %d, obj.: %f)" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i], d_obj[i])
        elif i < (len(d_rule)-1):
            print "else if %s, then prob. = %f (+: %d, -: %d, obj.: %f)" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i], d_obj[i])
        else:
            assert d_rule[i] == 0
            print "else prob. = %f (+: %d, -: %d, obj.: %f)" \
                % (d_prob[i], d_pos[i], d_neg[i], d_obj[i])
    print "objective = %f" % L_d

def write_rule_list(fname, d_rule, d_prob, ruleset, d_pos, d_neg, d_obj, L_d,
                    **options):
    append = True
    if "append" in options:
        append = options.get("append")
    
    if append:
        f = open(fname, 'a+')
    else:
        f = open(fname, 'w+')
    
    if "title" in options:
        f.write(options.get("title"))
    
    for i in range(len(d_rule)):
        if i == 0:
            f.write("if %s, then prob. = %f (+: %d, -: %d, obj.: %f)\n" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i], d_obj[i]))
        elif i < (len(d_rule)-1):
            f.write("else if %s, then prob. = %f (+: %d, -: %d, obj.: %f)\n" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i], d_obj[i]))
        else:
            assert d_rule[i] == 0
            f.write("else prob. = %f (+: %d, -: %d, obj.: %f)\n" \
                % (d_prob[i], d_pos[i], d_neg[i], d_obj[i]))
    f.write("objective = %f\n" % L_d)
    f.close()

def display_softFRL(d_rule, d_prob, ruleset,
                    d_pos, d_neg, d_pos_prop, d_obj, L_d):
    for i in range(len(d_rule)):
        if i == 0:
            print "if %s, then prob. = %f (+: %d, -: %d, +prop: %f, obj.: %f)" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i],
                   d_pos_prop[i], d_obj[i])
        elif i < (len(d_rule)-1):
            print "else if %s, then prob. = %f (+: %d, -: %d, +prop: %f, obj.: %f)" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i],
                   d_pos_prop[i], d_obj[i])
        else:
            assert d_rule[i] == 0
            print "else prob. = %f (+: %d, -: %d, +prop: %f, obj.: %f)" \
                % (d_prob[i], d_pos[i], d_neg[i], d_pos_prop[i], d_obj[i])
    print "objective = %f" % L_d

def write_softFRL(fname, d_rule, d_prob, ruleset,
                  d_pos, d_neg, d_pos_prop, d_obj, L_d, **options):
    append = True
    if "append" in options:
        append = options.get("append")
    
    if append:
        f = open(fname, 'a+')
    else:
        f = open(fname, 'w+')
    
    if "title" in options:
        f.write(options.get("title"))
    
    for i in range(len(d_rule)):
        if i == 0:
            f.write("if %s, then prob. = %f (+: %d, -: %d, +prop: %f, obj.: %f)" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i],
                   d_pos_prop[i], d_obj[i]))
        elif i < (len(d_rule)-1):
            f.write("else if %s, then prob. = %f (+: %d, -: %d, +prop: %f, obj.: %f)" \
                % (ruleset[d_rule[i]], d_prob[i], d_pos[i], d_neg[i],
                   d_pos_prop[i], d_obj[i]))
        else:
            assert d_rule[i] == 0
            f.write("else prob. = %f (+: %d, -: %d, +prop: %f, obj.: %f)" \
                % (d_prob[i], d_pos[i], d_neg[i], d_pos_prop[i], d_obj[i]))
    f.write("objective = %f\n" % L_d)
    f.close()
