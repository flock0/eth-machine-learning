import os
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing

def write_submission_multi_class(Y):
    '''
    Use this function if you are doing multi-class classif

    The Label column repeatedly contains the three label titles: "gender, "age", "health" (it is recommended to preserve this order).
    The last column, which is entitled Predicted, is your predicted value for the label. The predicted value is either TRUE or FALSE
    for all labels, where TRUE indicates female for the gender, young for the age, or healthy for the health status.
    For example, if you predict the test sample indexed by 0 as male, young and sick, then this prediction has to be written as:

    0 0 gender FALSE
    1 0 age      TRUE
    2 0 health  FALSE
    '''
    with open("./submission.csv", "w") as fw:
        fw.write('"ID","Sample","Label","Predicted"\n')
        i = 0
        label_type = ['gender', 'age', 'health']
        for y in Y:
            print y
            y = int(y)
            label = [y & 4, y & 2, y & 1]
            print label
            for y_j in label:
                fw.write("{},{},{},{}\n".format(i, i//3, label_type[i%3], 'TRUE' if y_j else 'FALSE'))
                i += 1

def write_submission_multi_label(Y):
    '''
    Use this when Y has one col for each of the labels

    0 0 gender FALSE
    1 0 age      TRUE
    2 0 health  FALSE
    '''
    with open("./final_sub.csv", "w") as fw:
        fw.write('"ID","Sample","Label","Predicted"\n')
        i = 0
        label_type = ['gender', 'age', 'health']
        for y in Y:
            print y
            #label = [y & 4, y & 2, y & 1]
            #print label
            for y_j in y:
                fw.write("{},{},{},{}\n".format(i, i//3, label_type[i%3], 'TRUE' if y_j else 'FALSE'))
                i += 1

def read_label_multi_class(filename='./targets.csv'):
    '''
    <male(0)/female(1)> <young(1)/old(0)> <sick(0)/healthy(1)>
    1,1,1 -> 7
    1,1,0 -> 6 - no female young sick
    1,0,1 -> 5
    1,0,0 -> 4
    0,1,1 -> 3
    0,1,0 -> 2 - no male young sick
    0,0,1 -> 1
    0,0,0 -> 0
    '''
    Y = np.loadtxt(filename, delimiter=',', dtype = int)
    # print Y
    # Z = np.empty([Y.shape[0]])
    # for i in range(Y.shape[0]):
    #     print "Y[i]: " + str(Y[i])
    #     t1 = (Y[i] == np.array([1,1,1]))
    #     t2 = (Y[i] == np.array([1,1,0]))
    #     t3 = (Y[i] == np.array([1,0,1]))
    #     t4 = (Y[i] == np.array([1,0,0]))
    #     t5 = (Y[i] == np.array([0,1,1]))
    #     t6 = (Y[i] == np.array([0,1,0]))
    #     t7 = (Y[i] == np.array([0,0,1]))
    #     t8 = (Y[i] == np.array([0,0,0]))
    #     if t1[0] and t1[1] and t1[2]:
    #         Z[i] = 7
    #     if t2[0] and t2[1] and t2[2]:
    #         Z[i] = 6
    #     if t3[0] and t3[1] and t3[2]:
    #         Z[i] = 5
    #     if t4[0] and t4[1] and t4[2]:
    #         Z[i] = 4
    #     if t5[0] and t5[1] and t5[2]:
    #         Z[i] = 3
    #     if t6[0] and t6[1] and t6[2]:
    #         Z[i] = 2
    #     if t7[0] and t7[1] and t7[2]:
    #         Z[i] = 1
    #     if t8[0] and t8[1] and t8[2]:
    #         Z[i] = 0
    #     print "Z[i] " + str(Z[i])

    return np.array(list(map(lambda y: 4 * y[0] + 2 * y[1] + y[2], Y)))

def read_label_multi_label(filename='./targets.csv'):
    return np.loadtxt(filename, delimiter=',', dtype = int)

def read_reduce(features, labels, TRAIN_SIZE):
    """
    features - list with feature files
    labels - numpy array with labels
    """
    total = np.ones((416,1)) # will be removed when testing std variation
    for name in features:
        f = np.load(name)
        print("shape of {0}, {1}".format(name, f.shape))
        total = np.append(total, f, axis=1)
    features_standardized = preprocessing.scale(total)
    print("Removing features which do not vary...")
    print(features_standardized.shape)
    features_standardized = features_standardized[:, np.std(features_standardized, axis=0) != 0]
    print(features_standardized.shape)
    selection = SelectKBest(score_func=f_classif, k=1000).fit(features_standardized[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    features_reduced = selection.transform(features_standardized)
    print("shape of reduced features:", features_reduced.shape)
    return features_reduced

def vote(labels):
    """
    Picks the class with the highest number of votes

    labels - numpy array
    """
    x = np.empty([pred.shape[0]])
    for i in range(pred.shape[0]):
        #x[i] = int(round(np.mean(labels[i,:])))
        x[i] = int(stats.mode(pred[i,:])[0])
    return x

if __name__ == "__main__":
    print(read_label_multi_class())
    print(read_label_multi_label())