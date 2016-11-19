#!/usr/bin/env python

"""Predict the playing strength of a chess player based on commentary on their games.

Implemented using scikit learn. Strength is predicted either by binary
classification (above or below mean?) using an SVM (linear or RBF
kernel), ranking (SVMRank) or regression.

Author: Christian Scheible <scheibcn@ims.uni-stuttgart.de>

"""

import sklearn
from sklearn import svm, linear_model, tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation

from util import *
from sys import stdin, stderr, stdout
from collections import Counter
import numpy
import scipy.sparse as sp
from scipy.stats import mstats, spearmanr, kendalltau
from ranking import *
import random
    

### hyperparameters

# model
model = "svm"  # svm or lr

# binning
n_bins = 2
n_folds = 10
manual_bins = [0, 1300, 1500, 3000]  # beginner middle, expert
manual_bins = [0, 1450, 3000]        # below average, above average
manual_bins = [0, 1700, 3000]
manual_bins = [  673.,    1062.75,  1452.5,  2232.  ]   # beginner, avg, expert
use_manual_bins = False
ranker_use_ratings = True

# feature transformations
binarize = False
scale_ratings = True
normalize_input = True
scale_input = False
sqrt_transform = False

# feature engineering
remove_stopwords = True
bigrams = True
lexical_features = False
master_sim = True
rule_features = True
length_feature = False
min_feature_freq = 5

# data for feature extraction
feature_file

= "chess-terms-small.txt"
master_files = ["masters/famous_games.pgn"]

lex = LexiconFeatures(feature_file)


### pre-processing
stdout.write("Reading games\n")

data = open("pgn_dumps/first100.pgnx").read()

games = read_pgnx(data)
random.seed(1234)
random.shuffle(games)
ratings = [game[1] for game in games]
annotations = [game[3] for game in games]
annotation_strings = [" ".join(a) for a in annotations]

n_games = len(games)


# tokenization
stdout.write("Tokenizing\n")
token_lists = [tokenize_annotations(a, binarize=binarize, nostop=remove_stopwords, bigrams=bigrams, special_features=rule_features) for a in annotation_strings]
annotation_lengths = [len(t) for t in token_lists]


# word counts
stdout.write("Counting\n")
token_dicts = [dict(Counter(tl)) for tl in token_lists]
vocabulary = set.union(*[set(tl) for tl in token_lists])


# filtering
filter_dicts(token_dicts, min_df=min_feature_freq)


# lexical features
if lexical_features:
    for i in xrange(n_games):
        lf = lex.count_terms(token_lists[i])
        token_dicts[i].update(lf)


# length feature
if length_feature:
    length_ratios = [len(annot.split())/float(len(annots)) for annot, annots in zip(annotation_strings,annotations)]
    norm = max(length_ratios)
    for i in xrange(n_games):
        token_dicts[i]["SPECIAL:len"] = length_ratios[i]/float(norm)


# vectorizing
stdout.write("Feature extraction\n")
vectorizer = DictVectorizer()
fv = vectorizer.fit_transform(token_dicts)


# scaling
if scale_input:
    fv = fv.todense()
    #sklearn.preprocessing.scale(fv, copy=False)
    scaler = sklearn.preprocessing.MinMaxScaler()
    fv = scaler.fit_transform(fv)


# normalization
if normalize_input:
    sklearn.preprocessing.normalize(fv, norm='l2', axis=1, copy=False)

if sqrt_transform:
    fv = sp.csr_matrix(numpy.sqrt(fv.todense()))


if not scale_input:
    fv = sp.csc_matrix(fv)


# master similarity feature
print "Loading master games"
masters = MasterSimilarity(master_files, vectorizer=vectorizer, binarize=binarize, nostop=remove_stopwords, bigrams=bigrams, special_features=rule_features)
sim = masters.get_centroid_sim(fv)
print "Master correlation:", spearmanr(ratings, numpy.array(sim.T)[0])

if master_sim:
    scaler_sim = sklearn.preprocessing.MinMaxScaler()
    fv =sp.csc_matrix( sp.hstack([fv, sim]))
    if normalize_input:
        sklearn.preprocessing.normalize(fv, norm='l2', axis=1, copy=False)
    vectorizer.feature_names_.append("MASTERSIM")

# rating statistics
rating_mean = numpy.mean(ratings)
rating_std = numpy.std(ratings)


# binning
stdout.write("Binning ratings\n")

if use_manual_bins:
    bin_counts, bins = numpy.histogram(ratings, manual_bins)
else:
    bin_counts, bins = numpy.histogram(ratings, n_bins)

bin_assignments = numpy.array([get_bin(bins, rating) for rating in ratings])
ratings = numpy.array(ratings, dtype=float)

original_ratings = ratings.copy()

if scale_ratings:
    sklearn.preprocessing.scale(ratings, copy=False)

print "Bins:", bins
print "Bin assignments:", bin_assignments




### SVM training
#rs = cross_validation.ShuffleSplit(n_games, n_iter=10, test_size=.25, random_state=0)
rs = cross_validation.KFold(n_games, n_folds=n_folds)


# storage for predictions
preds_SVM = []
preds_BL = []
golds = []
gold_orig_rating = []
rankings_SVM = []

# collecting numbers
accuracies_BL = []
f1_scores_BL = []
abs_rating_diffs_BL = []

accuracies_SVM = []
f1_scores_SVM = []
abs_rating_diffs_SVM = []

ranking_scores = []
ranking_taus = []
ranking_ys = []


### Experiments
# iterate over folds
for round_id, (train_index, test_index) in enumerate(rs):
    print "--------------- Fold", round_id, "----------------"

    ## SVM CLASSIFIER
    stdout.write("Training SVM classifier\n")
    
    X_train = fv[train_index,:]
    y_train = bin_assignments[train_index]

    if model == "svm":
        #clf = svm.SVC(kernel="linear") # , class_weight = {1:2}
        #clf = svm.SVC(kernel="rbf") # , class_weight = {1:2}
        clf = svm.LinearSVC()
    elif model == "lr":
        clf = linear_model.LogisticRegression()
    else:
        print "Unknown model type"
        assert False

    # training
    clf.fit(X_train, y_train)
    write_svm_features(clf, vectorizer, round=round_id)
    print "Classes:", clf.classes_

    X_test = fv[test_index,:]
    y_test = bin_assignments[test_index]

    golds.append(y_test)
    
    pred = clf.predict(X_test)

    preds_SVM.append(pred)

    # baseline
    most_frequent_value = mstats.mode(y_train)[0][0]
    majority_pred = numpy.repeat(most_frequent_value, y_test.shape[0])

    preds_BL.append(majority_pred)
    print pred

    # evaluation
    accuracy_BL  = sklearn.metrics.accuracy_score(y_test, majority_pred)
    accuracy_SVM = sklearn.metrics.accuracy_score(y_test, pred)

    accuracies_BL.append(accuracy_BL)
    accuracies_SVM.append(accuracy_SVM)

    f1_BL  = sklearn.metrics.f1_score(y_test, majority_pred, average="macro")
    f1_SVM = sklearn.metrics.f1_score(y_test, pred, average="macro")

    f1_scores_BL.append(f1_BL)
    f1_scores_SVM.append(f1_SVM)

    print "Majority BL accuracy:", accuracy_BL
    print "SVM accuracy:", accuracy_SVM
    
    print "Majority BL macro F1:", f1_BL
    print "SVM macro F1:", f1_SVM
    

    ## RANKING SVM
    orig_rating_train = original_ratings[train_index]
    orig_rating_test = original_ratings[test_index]
    gold_orig_rating.append(orig_rating_test)

    # use rank transformation or ratings?
    if ranker_use_ratings:
        ranker_y_train = orig_rating_train
        ranker_y_test = orig_rating_test
    else:
        ranker_y_train = y_train
        ranker_y_test = y_test

    # training
    rank_svm = RankSVM().fit(X_train.todense(), ranker_y_train)
    ranking_score = rank_svm.score(X_test.todense(), ranker_y_test)
    print 'Performance of ranking ', ranking_score

    #ranking = X_test.dot(rank_svm.coef_.ravel().T).flatten()
    ranking = rank_svm.decision_function(X_test.todense())

    # evaluation
    try:
        rho = spearmanr(ranking, ranker_y_test)
    except:
        # probably broke because of transposed matrix. undo and retry
        ranking = np.array(ranking.T)[0]
        rho = spearmanr(ranking, ranker_y_test)
        
    try:
        write_svm_features(rank_svm, vectorizer, round=round_id, filename="ranking-features")
    except:
        print "Could not write features (using RBF kernel?)"
    
    rankings_SVM.append(ranking)
    ranking_ys.append(ranker_y_test)
    ranking_scores.append(ranking_score)
    ranking_taus.append(rho)

    
    print "Spearman R results:", rho
    



    ## SVM REGRESSION
    stdout.write("Training SVM regressor\n")

    y_reg_train = ratings[train_index]

    # rbf kernel, elo has normal distribution
    reg = svm.SVR(kernel="rbf") # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    
    # training
    reg.fit(X_train, y_reg_train)
    
    y_reg_test = ratings[test_index]
    pred_reg = reg.predict(X_test)

    print y_reg_test
    print pred_reg

    # evaluation
    absolute_rating_diffs_BL  = abs(y_reg_test - numpy.repeat(numpy.mean(y_reg_train), y_reg_test.shape[0]))
    absolute_rating_diffs_SVM = abs(y_reg_test - pred_reg)

    mean_abs_rating_diff_BL  = numpy.mean(absolute_rating_diffs_BL)
    mean_abs_rating_diff_SVM = numpy.mean(absolute_rating_diffs_SVM)

    abs_rating_diffs_BL.append(mean_abs_rating_diff_BL)
    abs_rating_diffs_SVM.append(mean_abs_rating_diff_SVM)


    print "Mean abs rating diff BL:", mean_abs_rating_diff_BL
    print "Mean abs rating diff SVM:", mean_abs_rating_diff_SVM 



### print results
most_frequent_value = mstats.mode(bin_assignments)[0][0]

overall_y = numpy.hstack(golds)
preds_BL = numpy.repeat(most_frequent_value, n_games)
preds_SVM = numpy.hstack(preds_SVM)
gold_orig_rating = numpy.hstack(gold_orig_rating)
rankings_SVM = numpy.hstack(rankings_SVM)
ranking_ys = numpy.hstack(ranking_ys)

tau = spearmanr(rankings_SVM, ranking_ys)


accuracy_BL  = sklearn.metrics.accuracy_score(overall_y, preds_BL)
accuracy_SVM = sklearn.metrics.accuracy_score(overall_y, preds_SVM)

f1_BL  = sklearn.metrics.f1_score(overall_y, preds_BL, average=None)
f1_SVM = sklearn.metrics.f1_score(overall_y, preds_SVM, average=None)

write_results(overall_y, preds_SVM, gold_orig_rating, annotation_strings)

print "-----------------------"
print "Model:", model
print "Ranker trained on ratings (rather than bins):", ranker_use_ratings
print "Ratings mean:", rating_mean
print "Ratings std:", rating_std
print "Bins:", bins
print "Bin counts:", bin_counts
print "k-folds:", n_folds
print "Binarized:", binarize
print "Scaled ratings:", scale_ratings
print "Removed stopwords:", remove_stopwords
print "Lexical features:", lexical_features
print "Special features:", rule_features
print "Master similarity:", master_sim
print "Normalized vectors:", normalize_input
print "Scaled vectors:", scale_input

print 

print '\033[1m'
print "Overall Acc BL:", accuracy_BL
print "Overall Acc SVM:", accuracy_SVM

print
print "Overall F1 BL:",  f1_BL, numpy.mean(f1_BL)
print "Overall F1 SVM:", f1_SVM, numpy.mean(f1_SVM)

print
print "Ranking Acc:", numpy.mean(ranking_scores)
print "Overall Spearman r:", tau
print "Mean Spearman r:", numpy.mean(ranking_taus)

print
print "Mean abs rating diff BL:", numpy.mean(abs_rating_diffs_BL)
print "Mean abs rating diff SVM:", numpy.mean(abs_rating_diffs_SVM)

print '\033[0m'


#print "--- Report for BL ---"
#print sklearn.metrics.classification_report(overall_y, overall_pred_BL)

#print "--- Report for SVM ---"
#print sklearn.metrics.classification_report(overall_y, overall_pred_SVM)
