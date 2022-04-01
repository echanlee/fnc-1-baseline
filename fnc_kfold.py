import sys
import numpy as np
# import tensorflow as tf
from sklearn import feature_extraction
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import initialize_test, refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# from keras.models import Sequential
# from keras.preprocessing.text import Tokenizer
# from keras.layers import Dense, Embedding, Dropout, Flatten
# from keras import regularizers

from utils.system import parse_params, check_version

stop_words = feature_extraction.text.ENGLISH_STOP_WORDS
BATCH_SIZE = 128

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
    
def generate_tfidf(stances, dataset):
    heads, bodies = [],[]

    for stance in stances:
        heads.append(stance['Headline'])
        bodies.append(dataset.articles[stance['Body ID']])
    bow_vectorizer = CountVectorizer(max_features=5000, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words).\
        fit(heads + bodies ) 
    return bow_vectorizer, tfreq_vectorizer, tfreq, tfidf_vectorizer

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    bow_vectorizer, tfreq_vectorizer, tfreq, tfidf_vectorizer = generate_tfidf(d.stances, d)
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = initialize_test(competition_dataset.stances, competition_dataset, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = initialize_test(hold_out_stances,d, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    for fold in fold_stances:
        Xs[fold],ys[fold] = initialize_test(fold_stances[fold],d, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


    best_score = 0
    best_fold = None
    # tokenizer = Tokenizer(num_words=115550, filters='!"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n') 
    # tokenizer.fit_on_texts(d.articles+competition_dataset.articles)

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]
        
        # clf = Sequential()
        # clf.add(Embedding(input_dim=len(tokenizer.word_index)+1,
        #                   output_dim=100,trainable=False, name='word_embedding_layer', 
        #                   mask_zero=True))
        # clf.add(Flatten())
        # clf.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        #     bias_regularizer=regularizers.l2(1e-4), name='hidden_layer',input_shape =(BATCH_SIZE,) ))
        # clf.add(Dropout(rate=0.20, name='dropout_1')) 

        # clf.add(Dense(2, activation='softmax', name='output_layer'))

        # clf.compile(loss='binary_crossentropy', optimizer='adam',
        #     metrics=['accuracy'])
        
        # clf.fit(X_train, y_train,
        # batch_size=BATCH_SIZE,
        # epochs=12,)
        

        # predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        # actual = [LABELS[int(a)] for a in y_test]

        # fold_score, _ = score_submission(actual, predicted)
        # max_fold_score, _ = score_submission(actual, actual)

        # score = fold_score/max_fold_score

        # print("Score for fold "+ str(fold) + " was - " + str(score))
        # if score > best_score:
        #     best_score = score
        #     best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    # actual = [LABELS[int(a)] for a in y_holdout]

    # print("Scores on the dev set")
    # report_score(actual,predicted)
    # print("")
    # print("")

    # #Run on competition dataset
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    # actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    # report_score(actual,predicted)
