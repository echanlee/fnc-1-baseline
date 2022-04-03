import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_engineering import clean, get_tokenized_lemmas, refuting_features, polarity_features, hand_features, gen_or_load_feats, remove_stopwords
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

def initialize_dataset(dataset, stances):
    X, y = [], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))

        value = "".join(stance['Headline']) + dataset.articles[stance['Body ID']]
        clean_value = clean(value)
        clean_value = get_tokenized_lemmas(clean_value)
        clean_value = remove_stopwords(clean_value)
        # print(clean_value)
        X.append("".join(clean_value))
    return X,y

def generate_values(values, tf_idf_transformer):
    count_vect = count_vectorizer.transform(values)
    X = tf_idf_transformer.transform(count_vect).toarray()

    return X

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    tf_idf_transformer = TfidfTransformer(smooth_idf=False)
    #Load the training dataset and generate folds
    d = DataSet()
    d_test = DataSet("test")
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    X,y_train = initialize_dataset(d, d.stances)
    count_vectorizer.fit(X)
    count_vect_train = count_vectorizer.transform(X)
    tf_idf_transformer.fit(count_vect_train)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_comp, y_competition = initialize_dataset(competition_dataset, competition_dataset.stances)
    X_competition = generate_values(X_comp, tf_idf_transformer)

    # Xs = dict()
    # ys = dict()

    # x_holdout, y_holdout = initialize_dataset(d, hold_out_stances)
    # # Load/Precompute all features now
    # X_holdout = generate_values(x_holdout,tf_idf_transformer)
    # for fold in fold_stances:
    #     temp,ys[fold] = initialize_dataset(d,hold_out_stances)
    #     Xs[fold] = generate_values(temp,tf_idf_transformer)
    X_train = generate_values(X,tf_idf_transformer)
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(X_train, y_train)
    best_score = 0
    best_fold = None


    # Classifier for each fold
    # for fold in fold_stances:
    #     ids = list(range(len(folds)))
    #     del ids[fold]

    #     X_train = np.vstack(tuple([Xs[i] for i in ids]))
    #     y_train = np.hstack(tuple([ys[i] for i in ids]))

    #     X_test = Xs[fold]
    #     y_test = ys[fold]

    #     clf = LogisticRegression()
    #     clf.fit(X_train, y_train)

    #     predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
    #     actual = [LABELS[int(a)] for a in y_test]

    #     fold_score, _ = score_submission(actual, predicted)
    #     max_fold_score, _ = score_submission(actual, actual)

    #     score = fold_score/max_fold_score

    #     print("Score for fold "+ str(fold) + " was - " + str(score))
    #     if score > best_score:
    #         best_score = score
    #         best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    # actual = [LABELS[int(a)] for a in y_holdout]

    # print("Scores on the dev set")
    # report_score(actual,predicted)
    # print("")
    # print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in clf.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
