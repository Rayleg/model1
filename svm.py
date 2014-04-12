
__author__ = "vk24"

import pylab as pl
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import sklearn.feature_extraction

# This function is copy of Melnikov Andrey code (trees.py)
def read_data_set(columns, ds_path='data/data.csv', with_targets=True):
    """ Parse CSV-file, choose no-None data

    :param ds_path: CSV-file path
    :param columns: returned from choose_targets
    :return: x(n x m), y(n x 1), feature_names
    """
    data = []
    targets = []
    with open(ds_path, 'rU') as ds_file:
        for line in ds_file:
            items = line.strip().split('\t')

            row = {}
            target = None
            row_valid = True
            for name, (column, parse_fun) in columns.iteritems():
                value = parse_fun(items[int(column) - 1])
                if value is None:
                    row_valid = False
                    break
                if name == "target":
                    target = value
                else:
                    row[name] = value

            if with_targets:
                if row_valid and row and target:
                    data.append(row)
                    targets.append(target)
            else:
                # For clf.predict()
                if row_valid and row:
                    data.append(row)

    dv = sklearn.feature_extraction.DictVectorizer()
    return dv.fit_transform(data).todense(), np.array(targets), dv.get_feature_names()


# This function is copy of Melnikov Andrey code (trees.py)
def parse_int(s):
    return int(s) if s != "-1" else None


def choose_targets():
    """
    :return: chosen columns for classification
    """
    columns = {
        #"relation": (4, parse_int),
        #"status_len": (5, parse_int), # B
        #"emotions": (6, parse_int),
        #"videos": (7, parse_int),
        #"notes": (8, parse_int),
        #"subscr": (9, parse_int),
        #"mut_friends": (10, parse_int),
        "audios": (11, parse_int), # B
        #"photos": (12, parse_int), 
        #"followers": (13, parse_int),
        #"albums": (14, parse_int),
        "friends": (15, parse_int), # B
        #"pages": (16, parse_int),
        #"likes": (17, parse_int),
        #"grad_year": (18, parse_int),
        #"activity": (19, parse_int),
        #"u_photos": (20, parse_int),
        "groups": (22, parse_int),
        "target": (2, parse_int)
    }
    return columns


def main():

    print "Hello!\n"
    print "Read data set\n"
    x, y, feature_names = read_data_set(choose_targets())

    C = 1.0
    folds = 3
    svc = svm.SVC(kernel='linear', C=C)                 # Very long
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)     # Ok
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C)    # Very long
    lin_svc = svm.LinearSVC(C=C)                        # Very long
    print "Start cross validation\n"
    scores = cross_val_score(rbf_svc, x, y, cv=folds)
    print "Rbf SVC Model mean accuracy: {}".format(np.mean(scores))
    print "Features: {}".format(feature_names)
    return


if __name__ == "__main__":
    main()

