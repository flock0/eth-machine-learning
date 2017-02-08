import sys
import numpy as np

from sklearn import datasets
#from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss, hamming_loss, make_scorer, accuracy_score
from sklearn.decomposition import PCA, KernelPCA
from util import *

def s_error(scores, best_score):
    for i in range(len(scores)):
        if(best_score == scores[i][1]):
            return np.sqrt(np.var(scores[i][2]))

ham_scorer = make_scorer(hamming_loss)

if __name__=='__main__':
    TRAIN_SIZE = 278
    TEST_SIZE = 0
    FEATURES_FILE = sys.argv[1]
    print("Reading labels...")
    labels = np.concatenate([np.loadtxt("./data/targets.csv", delimiter=","), np.full((TEST_SIZE, 3), -1, dtype=int)])
    labels = labels[:,2]
    print "health"
    print labels.shape
    print("Reading feature matrix...")
    features = np.load(FEATURES_FILE)
    print("Removing features which do not vary...")
    print features.shape
    print np.std(features, axis = 0) == 0
    #features = features[:, np.std(features, axis=0) != 0]
    print features.shape
    print labels
    FOLDS = 10
    NJOBS = -1

    # Models
    lr = LogisticRegression(class_weight='balanced', multi_class = 'ovr')
    svm = SVC(probability=True)
    svmPoly = SVC(kernel='poly', probability=True)
    rf = RandomForestClassifier(class_weight='balanced')
    gbc = GradientBoostingClassifier()
    #kn = KNeighborsClassifier()
    #gp = GaussianProcessClassifier(copy_X_train=False)
    #qda = QuadraticDiscriminantAnalysis()

    # Setup pipeline
    union = FeatureUnion([('kernelpca', KernelPCA()), ('perc', SelectPercentile())])
    pipeLR = Pipeline([('featureunion', union), ('lr', lr)])
    pipeSVM = Pipeline([('featureunion', union), ('svm', svm)])
    # pipeSVMPoly = Pipeline([('featureunion', union), ('svmPoly', svmPoly)])
    pipeRF = Pipeline([('featureunion', union), ('rf', rf)])
    pipeGB = Pipeline([('featureunion', union), ('gbc', gbc)])
    # pipeKN = Pipeline([('featureunion', union), ('kn', kn)])
    # pipeGP = Pipeline([('featureunion', union), ('gp', gp)])
    # pipeQDA = Pipeline([('featureunion', union), ('qda', qda)])

    # Search space for the parameter tuning
    kernelPCAKernel = ['linear', 'rbf', 'poly']
    percentile = np.array([18, 20, 22])
    scoreFunc = [f_classif]
    lrC = np.array([50, 100, 300, 500])
    svmC = np.array([0.0001, 0.001, 0.1, 0.25])
    svmGamma = np.array([0.1, 0.05, 0.01, 0.005])
    svmKernel = ['linear', 'rbf']
    # svmPolyC = np.logspace(0, 4, num=5, base=10)
    # svmPolyGamma = np.array([0.01, 0.001, 0.0001])
    # svmPolyDegree = np.array([2, 3, 5])
    rfNEstimators = np.array([50, 100, 200, 400])
    rfMinSamplesSplit = np.array([2, 3, 4])
    gbLoss = ["deviance", "exponential"]
    gbMaxDepth = np.array([3,5,7,9])
    gbLearningRate = np.array([0.1,0.5,1.0])
    gbNEstimators = np.array([100,300,500,700,900,1100,1300,1500])
    # knNumNeighbors = [3, 5, 10]
    # knWeights = ['uniform', 'distance']
    # gpMaxIterPredict = np.array([50, 100, 200])
    # qdaRegParam = np.array([0.0])

    best_scores = []

    # print("Multiclass logistic regression...")
    # grid = GridSearchCV(estimator=pipeLR, param_grid=dict(lr__C=lrC, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__perc__percentile=percentile, featureunion__perc__score_func=scoreFunc), scoring="accuracy", cv=FOLDS)
    # grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    # best_scores.append(["LR", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])

    print("SVM...")
    grid = GridSearchCV(estimator=pipeSVM, param_grid=dict(svm__C=svmC, svm__gamma=svmGamma, svm__kernel=svmKernel, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__perc__percentile=percentile, featureunion__perc__score_func=scoreFunc), scoring='accuracy', cv=FOLDS)
    grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    best_scores.append(["SVM", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])

    # print("Random forest...")
    # grid = GridSearchCV(estimator=pipeRF, param_grid=dict(rf__n_estimators=rfNEstimators, rf__min_samples_split=rfMinSamplesSplit, featureunion__kernelpca__kernel=kernelPCAKernel,featureunion__perc__percentile=percentile, featureunion__perc__score_func=scoreFunc), scoring='accuracy', cv=FOLDS)
    # grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    # best_scores.append(["RF", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])


    # print("GBC...")
    # grid = GridSearchCV(estimator=pipeGB, param_grid=dict(gbc__loss=gbLoss, gbc__max_depth=gbMaxDepth,gbc__learning_rate=gbLearningRate, gbc__n_estimators=gbNEstimators, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest, featureunion__kbest__score_func=kBestScoreFunc), scoring="accuracy", cv=FOLDS)
    # grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    # best_scores.append(["GBC", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])

    # print("KN...")
    # grid = GridSearchCV(estimator=pipeKN, param_grid=dict(kn__n_neighbors=knNumNeighbors, kn__weights=knWeights, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest, featureunion__kbest__score_func=kBestScoreFunc), scoring="accuracy", cv=FOLDS)
    # grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    # best_scores.append(["KN", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])
    #
    # print("GaussianProcess...")
    # grid = GridSearchCV(estimator=pipeGP, param_grid=dict(gp__max_iter_predict=gpMaxIterPredict, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest, featureunion__kbest__score_func=kBestScoreFunc), scoring="accuracy", cv=FOLDS)
    # grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    # best_scores.append(["GP", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])
    #
    # print("QuadraticDiscriminantAnalysis...")
    # grid = GridSearchCV(estimator=pipeQDA, param_grid=dict(qda__reg_param=qdaRegParam, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest, featureunion__kbest__score_func=kBestScoreFunc), scoring="accuracy", cv=FOLDS)
    # grid.fit(features[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
    # best_scores.append(["QDA", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_), grid.best_params_])

    # Overview of all models
    bestScoresArray = np.array(best_scores)
    print best_scores
    print("Best model with optimal parameter", bestScoresArray[np.argmax(bestScoresArray[:, 1])])
