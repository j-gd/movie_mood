import numpy as np
import pandas as pd
import pickle
import math
import datetime
from IPython.display import Markdown, display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# From this project
from NLPv0 import WordBag, AboutMovie

class StarRater():
    def __init__(self):
        # Subsampling from Amazon reviews
        # self.NB_SAMPLES = 360000  #4000  # up to 200k, then change the input file

        self.data_path = '../../datasets/'
        self.xl_report = 'gbc_500_trees_02_rate_8_depth_5_leaf_sqrt_10k_tfidf.xlsx'

        # TIDF setup
        MAX_FEATURES = 10000

        self.tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            max_features=MAX_FEATURES,
            norm='l2',  # normalize each review
            use_idf=True)


        N_TREES = 500
        LEARN_RATE = 0.2
        MAX_DEPTH = 8
        MIN_IN_LEAF = 5  #7
        MAX_FEATURES = 'sqrt'

        self.gbc = GradientBoostingClassifier(learning_rate=LEARN_RATE,
                                        n_estimators=N_TREES,
                                        min_samples_leaf=MIN_IN_LEAF,
                                        max_depth=MAX_DEPTH,
                                        max_features=MAX_FEATURES)

        self.report = {
            'case': [],
            'remove': [],
            'percent': [],
            'in_train_p': [],
            'in_train_n': [],
            'in_test_p': [],
            'in_test_n': [],
            'in_total': [],
            'xy_check': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
        }


    def predict_and_report(
            self,
            case,
            report_file,
            remove='None',
            pct='0',
    ):
        print('Starting case {} {} {}'.format(case, remove, pct))
        print (str(datetime.datetime.now()))
        self.report['case'].append(case)
        self.report['remove'].append(remove)
        self.report['percent'].append(pct)
        root = 'reviews_wout_top_' + pct + 'pct_' + remove
        if case == 'A':
            pickle_in = open(self.data_path + 'reviews_A.pkl', "rb")
            movie_reviews = pickle.load(pickle_in)
        else:
            pickle_in = open(self.data_path + root + '_B.pkl', "rb")
            movie_reviews = pickle.load(pickle_in)
        pickle_in.close()

        for i in ['train','test']:
            for j in ['positive','negative']:
                movie_reviews[i][j] = movie_reviews[i][j].rename(columns={'sentence':'reviewText'})

        total = 0
        for i in ['train','test']:
            for j in ['positive','negative']:
                total += movie_reviews[i][j].shape[0]
        self.report['in_train_p'].append(movie_reviews['train']['positive'].shape[0])
        self.report['in_train_n'].append(movie_reviews['train']['negative'].shape[0])
        self.report['in_test_p'].append(movie_reviews['test']['positive'].shape[0])
        self.report['in_test_n'].append(movie_reviews['test']['negative'].shape[0])
        self.report['in_total'].append(total)

        train_words = pd.concat([movie_reviews['train']['positive']['reviewText'],
                            movie_reviews['train']['negative']['reviewText']])
        y_train = np.concatenate([np.ones((movie_reviews['train']['positive'].shape[0],)),
                                  np.zeros((movie_reviews['train']['negative'].shape[0],))])
        test_words = pd.concat([movie_reviews['test']['positive']['reviewText'],
                            movie_reviews['test']['negative']['reviewText']])
        y_test = np.concatenate([np.ones((movie_reviews['test']['positive'].shape[0],)),
                                  np.zeros((movie_reviews['test']['negative'].shape[0],))])

        SPARSE = True

        if SPARSE:
            # Optimization: add the review length while keeping sparse matrix
            tf_train = self.tfidf.fit_transform(train_words)
            tf_test = self.tfidf.transform(test_words)
        else:
            tf_train = self.tfidf.fit_transform(train_words).todense()
            tf_test = self.tfidf.transform(test_words).todense()

        # option: add length to input
        ADD_LENGTH = False

        if ADD_LENGTH:
            if SPARSE:
                # Hack: pick an existing word to store the count
                len_idx = 0
                test_lengths = [len(words) for words in test_words]

                for idx,words in enumerate(train_words):
                    tf_train[idx][len_idx] = len(words)
                for idx,words in enumerate(test_words):
                    tf_test[idx][len_idx] = len(words)
                X_train = tf_train
                X_test = tf_test
            else:
                train_lengths = np.array([len(words) for words in train_words]).reshape(-1,1)
                test_lengths = np.array([len(words) for words in test_words]).reshape(-1,1)
                X_train = np.concatenate([tf_train, train_lengths],axis=1)
                X_test = np.concatenate([tf_test, test_lengths],axis=1)
        else:
            X_train = tf_train
            X_test = tf_test

        if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
            self.report['xy_check'].append('problem!!!')
        else:
            self.report['xy_check'].append('OK')

        self.gbc.fit(X_train, y_train)

        y_pred = self.gbc.predict(X_test)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        self.report['precision'].append(prec)
        self.report['recall'].append(rec)
        self.report['f1'].append(f1)
        self.report['accuracy'].append(accuracy_score(y_test, y_pred, normalize=True))

        #             for key, val in self.report.items():
        #                 print(' ')
        #                 print(key)
        #                 print(val)

        pd.DataFrame(self.report).to_excel(xl_report)
