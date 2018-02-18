# data.py: data loading
# author: Chaofan Chen
#
import numpy as np

def load_data(fname_data, **options):
    # exclude_column is a valid option:
    # it should give the index of the column to be excluded;
    # the index starts at 0
    with open(fname_data, 'r') as f:
        instances = f.readlines()
    data = []
    for instance in instances:
        entry = instance.split()
        if "exclude_column" in options:
            del entry[options.get("exclude_column")]
            data.append(entry)
        else:
          data.append(entry)
    return data

def load_labels(fname_label, **options):
    # label_column is a valid option:
    # it should give the index of the column which gives the labels;
    # the index starts at 0
    Y = np.loadtxt(fname_label)    
    if "label_column" in options:
        return Y[:, options.get("label_column")]
    else:
        assert len(Y.shape) == 1
        return np.array(Y)

def load_testset(fname_test_data, fname_test_label, ruleset, **options):
    # exclude_column and label_column are valid options    
    
    # load the test data
    if "exclude_column" in options:
        test_data = load_data(fname_test_data,
                              exclude_column = options.get("exclude_column"))
    else:
        test_data = load_data(fname_test_data)
    
    if "label_column" in options:
        Ytest = load_labels(fname_test_label,
                            label_column = options.get("label_column"))
    else:
        Ytest = load_labels(fname_test_label)
    
    # form the rule-versus-test-data set
    # Xtest[j] is the set of test data points that satisfy rule j
    Xtest = [set() for j in range(len(ruleset)+1)]
    Xtest[0] = set(range(len(test_data)))    
    for (j, rule) in enumerate(ruleset):
        if j > 0:
            Xtest[j] = set([i for (i, xi) in enumerate(test_data) \
                            if set(rule).issubset(xi)])
    
    return test_data, Xtest, Ytest
