import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from IPython.display import Markdown, display
import pickle


class SubjectiveFilter():
    def __init__(self, path_to_fit_tfidf, path_to_fit_model):
        pickle_in = open(path_to_fit_tfidf, 'rb')
        self.obj_tfidf = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open(path_to_fit_model, 'rb')
        self.obj_model = pickle.load(pickle_in)
        pickle_in.close()

    def to_one_sent_per_row(self, df):
        df['sentence'] = df['reviewText'].map(sent_tokenize)

        # Create a dataframe with one line per sentence
        return df['sentence'] \
        .apply(pd.Series) \
        .merge(df, left_index = True, right_index = True) \
        .drop(['sentence','reviewText'], axis = 1) \
        .melt(id_vars = ['reviewerID', 'asin','overall'], value_name = 'sentence') \
        .drop(['variable'], axis = 1) \
        .dropna()


    def transform(self,
                  sentences,
                  text_col,
                  remove_fraction,
                  debug_level=0,
                  remove='obj'):  # -> pd.DataFrame:
        '''
        Input:
          df: Pandas dataframe with one sentence per row
          text_col: text column name

        Output:
          A pandas dataframe with objective sentences removed 
        '''

        # Figure out which sentences are subjective
        obj_X = self.obj_tfidf.transform(sentences[text_col]).todense()


        # Remove objective sentences
        if remove == 'obj':
            y_test = self.obj_model.predict_proba(obj_X)[:,1]
        else:
            y_test = self.obj_model.predict_proba(obj_X)[:,0]

        print('length',len(y_test))
        cutoff = int(round((1 - remove_fraction) * len(y_test)))
        print(cutoff)
        if cutoff == len(y_test):
          print('skipping this review, not enough sentences')
          return sentences, -1

        idx_keep = y_test.argsort()[:cutoff]
        print(idx_keep)
        subjective_sentences = sentences.iloc[idx_keep]

        display(Markdown('### Kept:'))
        display(subjective_sentences)

        display(Markdown('### Dropped:'))
        display( sentences.iloc[y_test.argsort()[-(len(y_test) - cutoff):]])

        nb_sentences_removed = sentences.shape[0] - subjective_sentences.shape[0]

        # Display # lines removed
        nb_sentences_removed = len(sentences) - len(subjective_sentences)
        display(Markdown('#### => Removed {0} ({1:.0%}) {2} sentences'.format(
                nb_sentences_removed, nb_sentences_removed / len(sentences), remove)))

        # Merge the objective sentences back into comments
        subj_groups = subjective_sentences.groupby(['reviewerID', 'asin'])
        subj_reviews_sentences = subj_groups[text_col].agg(
            self._merge_sentences)
        subj_reviews_stars = subj_groups['overall'].mean()
        subj_reviews = pd.merge(subj_reviews_sentences,
                        subj_reviews_stars,
                        how='inner', on=['reviewerID', 'asin']).reset_index()

        return subj_reviews, nb_sentences_removed  #, y_test

    def print_info(self, df, subj_reviews, debug_level=0):
        # Print info
        review_diff = df.shape[0] - subj_reviews.shape[0]
        display(Markdown('#### => Removed {0} ({1:.0%}) reviews with no emotional content'
                        .format(review_diff, review_diff / df.shape[0])))

        if debug_level > 0:
            # Check that the split and merge worked correctly
            df_groups = df.groupby(['reviewerID','asin'])
            df_stars = df_groups['overall'].mean()
            check = pd.merge(df_stars, subj_reviews,
                            how='inner', on=['reviewerID', 'asin'])
            res = (check['overall_x'] == check['overall_y']).mean()
            if res == 1:
                print('OK! Split-merge match')
            else:
                print('### @@@@@@@@ PROBLEM! @@@@@@')
                print('Split and merge stars differ')

            display(Markdown('##### A few objective sentences removed:'))
            display(self.objective_sentences[:debug_level+1])


    def _add_space(self, sentence):
        # Add space at the beginning of each sentence to help tokenizer recognize words
        return ' ' + sentence

    def _merge_sentences(self, series):
        return series.map(self._add_space).sum()
